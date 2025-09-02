// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fsst::{Compressor, Symbol};
use vortex_array::arrays::VarBinViewArray;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, ValidityHelper};
use vortex_array::{Array, Canonical, IntoArray, ProstMetadata};
use vortex_buffer::{Buffer, BufferMut, ByteBuffer, ByteBufferMut};
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{VortexResult, vortex_bail, vortex_ensure, vortex_err};

use crate::fsst_train_compressor;
use crate::fsst_view::{FSSTViewArray, FSSTViewEncoding, FSSTViewVTable, OutlinedStr, View};

#[derive(Clone, prost::Message)]
pub struct FSSTViewMetadata {
    #[prost(enumeration = "PType", tag = "1")]
    compressed_offsets_ptype: i32,
    #[prost(enumeration = "PType", tag = "2")]
    uncompressed_offsets_ptype: i32,
    #[prost(uint32, tag = "3")]
    offsets_len: u32,
}

impl SerdeVTable<FSSTViewVTable> for FSSTViewVTable {
    type Metadata = ProstMetadata<FSSTViewMetadata>;

    fn metadata(array: &FSSTViewArray) -> VortexResult<Option<Self::Metadata>> {
        let compressed_offsets_ptype = array.compressed_offsets.dtype().as_ptype();
        let uncompressed_offsets_ptype = array.uncompressed_offsets.dtype().as_ptype();
        let offsets_len = u32::try_from(array.compressed_offsets.len()).map_err(|_| {
            vortex_err!("FSSTViewArray should not contain >= 2^32 compressed strings")
        })?;

        Ok(Some(ProstMetadata(FSSTViewMetadata {
            compressed_offsets_ptype: compressed_offsets_ptype as i32,
            uncompressed_offsets_ptype: uncompressed_offsets_ptype as i32,
            offsets_len,
        })))
    }

    fn build(
        _: &FSSTViewEncoding,
        dtype: &DType,
        len: usize,
        metadata: &FSSTViewMetadata,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<FSSTViewArray> {
        // If the DType is nullable, we need to validate the validity information
        vortex_ensure!(
            dtype.is_utf8() || dtype.is_binary(),
            "FSSTViewArray can only be built for utf8 or binary data type, not {dtype}"
        );

        let [
            zstd_views_buffer,
            symbols_buffer,
            symbol_lengths_buffer,
            fsst_buffer,
        ] = buffers
        else {
            vortex_bail!("FSSTViewVTable: build requires exactly four buffers");
        };

        // Decode the views data.
        let expected_bytes = len * size_of::<View>();
        let mut views_buffer = ByteBufferMut::zeroed(expected_bytes);
        let decoded_bytes =
            zstd::bulk::decompress_to_buffer(zstd_views_buffer.as_slice(), &mut views_buffer)?;
        vortex_ensure!(
            expected_bytes == decoded_bytes,
            "ZSTD decoded {decoded_bytes} bytes, expected {expected_bytes} bytes"
        );

        let views = Buffer::<View>::from_byte_buffer(views_buffer.freeze());

        vortex_ensure!(
            views.len() == len,
            "FSSTViewArray: views expected to have length {len}, was {}",
            views.len()
        );

        // Second buffer are the symbol table
        let symbols = Buffer::<Symbol>::from_byte_buffer(symbols_buffer.clone());
        // Third buffer: symbol lengths
        let symbol_lengths = symbol_lengths_buffer.clone();

        // Fourth buffer: compressed strings
        let fsst_buffer = fsst_buffer.clone();

        vortex_ensure!(
            children.len() >= 2,
            "FSSTViewArray: must have 2 or more children"
        );

        let offsets_len = metadata.offsets_len as usize;
        let compressed_ptype = PType::try_from(metadata.compressed_offsets_ptype)
            .map_err(|err| vortex_err!("compressed_offsets_ptype enum value: {err}"))?;
        let uncompressed_ptype = PType::try_from(metadata.uncompressed_offsets_ptype)
            .map_err(|err| vortex_err!("uncompressed_offsets_ptype enum value: {err}"))?;

        let compressed_offsets = children
            .get(0, compressed_ptype.into(), offsets_len)
            .map_err(|err| vortex_err!("loading child[0]: compressed_offsets: {err}"))?;
        let uncompressed_offsets = children
            .get(1, uncompressed_ptype.into(), offsets_len)
            .map_err(|err| vortex_err!("loading child[1]: uncompressed_offsets: {err}"))?;

        let validity = if children.len() == 3 {
            let validity = children
                .get(2, &DType::Bool(Nullability::NonNullable), len)
                .map_err(|err| vortex_err!("loading child[0]: validity: {err}"))?;
            Validity::Array(validity)
        } else {
            Validity::from(dtype.nullability())
        };

        FSSTViewArray::try_new(
            views,
            fsst_buffer,
            symbols,
            symbol_lengths,
            compressed_offsets,
            uncompressed_offsets,
            dtype.clone(),
            validity,
        )
    }
}

impl EncodeVTable<FSSTViewVTable> for FSSTViewVTable {
    // Write into the encoding using the canonical elements.
    fn encode(
        _: &FSSTViewEncoding,
        canonical: &Canonical,
        like: Option<&FSSTViewArray>,
    ) -> VortexResult<Option<FSSTViewArray>> {
        let Canonical::VarBinView(strings) = canonical else {
            // Only VarBinView canonical types are supported.
            return Ok(None);
        };

        // Reuse the compressor from the other array to compress our array.
        match like {
            None => {
                let compressor = fsst_train_compressor(strings.as_ref())?;
                let symbols = compressor.symbol_table().iter().copied().collect();
                let symbol_lengths = compressor.symbol_lengths().iter().copied().collect();
                Ok(Some(compress_from_canonical(
                    strings,
                    &symbols,
                    &symbol_lengths,
                    &compressor,
                )))
            }
            Some(original) => Ok(Some(compress_from_canonical(
                strings,
                &original.symbols,
                &original.symbol_lengths,
                &original.compressor,
            ))),
        }
    }
}

/// Compress a canonical string array with FSST.
#[allow(clippy::cast_possible_truncation)]
fn compress_from_canonical(
    array: &VarBinViewArray,
    symbols: &Buffer<Symbol>,
    symbol_lengths: &ByteBuffer,
    compressor: &Compressor,
) -> FSSTViewArray {
    // Pre-allocate a reusable buffer for compression
    let mut reuse = Vec::with_capacity(16 * 1024 * 1024);
    let mut codes = ByteBufferMut::with_capacity(16 * 1024 * 1024);
    let mut uncompressed_offsets: BufferMut<u32> = BufferMut::with_capacity(array.views().len());
    let mut compressed_offsets: BufferMut<u32> = BufferMut::with_capacity(array.views().len());

    uncompressed_offsets.push(0);
    compressed_offsets.push(0);

    let mut views = BufferMut::with_capacity(array.views().len());
    let mut index = 0;

    for idx in 0..array.len() {
        let view = array.views()[idx];
        if view.is_inlined() {
            // Push a new uncompressed string view
            let inline_binary_view = view.as_inlined();
            views.push(View::new_inlined(
                &inline_binary_view.data[0..inline_binary_view.size as usize],
            ));

            continue;
        }

        // Compress and push outlined view pointer
        let view = view.as_view();
        let buffer = array.buffer(view.buffer_index as usize);
        let start = view.offset() as usize;
        let end = start + view.size as usize;

        // TODO(aduffy): handle strings larger than 8MB
        assert!(
            end - start < 8 * 1024 * 1024,
            "FSST cannot handle strings larger than 8MB at this time"
        );
        let uncompressed = &buffer[start..end];
        let uncompressed_len = uncompressed.len() as u32;
        unsafe { compressor.compress_into(uncompressed, &mut reuse) };

        codes.extend_from_slice(reuse.as_slice());

        let mut prefix = [0u8; 8];
        prefix.copy_from_slice(&uncompressed[0..8]);

        views.push(View {
            outline: OutlinedStr {
                len: uncompressed_len,
                index,
                prefix,
            },
        });

        index += 1;

        // We know reuse < 16MB, so should always fit in u32
        compressed_offsets
            .push(compressed_offsets.last().copied().unwrap_or(0) + reuse.len() as u32);
        // We know that plain string always < 8MB, so should fit comfortable in u32
        uncompressed_offsets
            .push(uncompressed_offsets.last().copied().unwrap_or(0) + uncompressed_len);
    }

    let uncompressed_offsets = uncompressed_offsets.freeze().into_array();
    let compressed_offsets = compressed_offsets.freeze().into_array();
    let views = views.freeze();

    // SAFETY: safe by construction
    unsafe {
        FSSTViewArray::new_unchecked(
            views,
            codes.freeze(),
            symbols.clone(),
            symbol_lengths.clone(),
            compressed_offsets,
            uncompressed_offsets,
            array.dtype().clone(),
            array.validity().clone(),
        )
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::arrays::builder::VarBinBuilder;
    use vortex_array::{ArrayRef, ToCanonical};
    use vortex_dtype::{DType, Nullability};

    use crate::fsst_view::FSSTViewEncoding;

    #[allow(clippy::unwrap_used)]
    fn build_simple_fsst_view_array() -> ArrayRef {
        let mut builder = VarBinBuilder::<i32>::with_capacity(3);
        builder.append_value(b"hello world");
        builder.append_value(b"this is a much longer string that exceeds the inline threshold");
        builder.append_value(b"short");

        let array = builder.finish(DType::Utf8(Nullability::NonNullable));

        FSSTViewEncoding
            .encode(&array.to_canonical(), None)
            .unwrap()
            .unwrap()
    }

    #[test]
    fn test_encode_canonicalize_roundtrip() {
        let fsst_view = build_simple_fsst_view_array();
        let round_trip_array = fsst_view.to_varbinview();

        // Verify same length
        assert_eq!(3, round_trip_array.len());

        // Test that we can get scalars without panic
        for i in 0..3 {
            let scalar = round_trip_array.scalar_at(i);
            assert!(!scalar.is_null());
        }
    }

    #[test]
    fn test_empty_strings() {
        let mut builder = VarBinBuilder::<i32>::with_capacity(2);
        builder.append_value(b"");
        builder.append_value(b"non-empty");

        let array = builder.finish(DType::Utf8(Nullability::NonNullable));

        let canonical = array.to_canonical();
        let fsst_view = FSSTViewEncoding.encode(&canonical, None).unwrap().unwrap();

        let round_trip_array = fsst_view.to_varbinview();

        assert_eq!(2, round_trip_array.len());

        // Verify scalars can be accessed
        for i in 0..2 {
            let scalar = round_trip_array.scalar_at(i);
            assert!(!scalar.is_null());
        }
    }

    #[test]
    fn test_mixed_nulls_array() {
        let mut builder = VarBinBuilder::<i32>::with_capacity(4);
        builder.append_value(b"first");
        builder.append_null();
        builder.append_value(b"second");
        builder.append_null();

        let array = builder.finish(DType::Utf8(Nullability::Nullable));

        let canonical = array.to_canonical();
        let fsst_view = FSSTViewEncoding.encode(&canonical, None).unwrap().unwrap();

        let round_trip_array = fsst_view.to_varbinview();

        assert_eq!(4, round_trip_array.len());

        // Check that nulls are preserved
        assert!(!round_trip_array.scalar_at(0).is_null());
        assert!(round_trip_array.scalar_at(1).is_null());
        assert!(!round_trip_array.scalar_at(2).is_null());
        assert!(round_trip_array.scalar_at(3).is_null());
    }

    #[test]
    fn test_binary_data() {
        let mut builder = VarBinBuilder::<i32>::with_capacity(2);
        builder.append_value([0u8, 1, 2, 3]);
        builder.append_value([255u8, 254, 253]);

        let array = builder.finish(DType::Binary(Nullability::NonNullable));

        let canonical = array.to_canonical();
        let fsst_view = FSSTViewEncoding.encode(&canonical, None).unwrap().unwrap();

        let round_trip_array = fsst_view.to_varbinview();

        assert_eq!(2, round_trip_array.len());

        // Verify scalars can be accessed
        for i in 0..2 {
            let scalar = round_trip_array.scalar_at(i);
            assert!(!scalar.is_null());
        }
    }

    #[test]
    fn test_single_element() {
        let mut builder = VarBinBuilder::<i32>::with_capacity(1);
        builder.append_value(b"single");

        let array = builder.finish(DType::Utf8(Nullability::NonNullable));

        let canonical = array.to_canonical();
        let fsst_view = FSSTViewEncoding.encode(&canonical, None).unwrap().unwrap();

        let round_trip_array = fsst_view.to_varbinview();

        assert_eq!(1, round_trip_array.len());
        assert!(!round_trip_array.scalar_at(0).is_null());
    }

    #[test]
    fn test_all_nulls() {
        let mut builder = VarBinBuilder::<i32>::with_capacity(3);
        builder.append_null();
        builder.append_null();
        builder.append_null();

        let array = builder.finish(DType::Utf8(Nullability::Nullable));

        let canonical = array.to_canonical();
        let fsst_view = FSSTViewEncoding.encode(&canonical, None).unwrap().unwrap();

        let round_trip_array = fsst_view.to_varbinview();

        assert_eq!(3, round_trip_array.len());

        // All should be null
        for i in 0..3 {
            assert!(round_trip_array.scalar_at(i).is_null());
        }
    }
}
