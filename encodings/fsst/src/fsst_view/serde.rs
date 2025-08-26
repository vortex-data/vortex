// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use fsst::{Compressor, Symbol};
use vortex_array::arrays::VarBinViewArray;
use vortex_array::serde::ArrayChildren;
use vortex_array::validity::Validity;
use vortex_array::vtable::{EncodeVTable, SerdeVTable, ValidityHelper};
use vortex_array::{Canonical, EmptyMetadata, IntoArray};
use vortex_buffer::{Buffer, BufferMut, ByteBuffer, ByteBufferMut};
use vortex_dtype::{DType, Nullability, PType};
use vortex_error::{vortex_bail, vortex_ensure, VortexExpect, VortexResult};

use crate::fsst_train_compressor;
use crate::fsst_view::{FSSTViewArray, FSSTViewEncoding, FSSTViewVTable, OutlinedStr, View};

impl SerdeVTable<FSSTViewVTable> for FSSTViewVTable {
    type Metadata = EmptyMetadata;

    fn metadata(_: &FSSTViewArray) -> VortexResult<Option<Self::Metadata>> {
        Ok(Some(EmptyMetadata))
    }

    fn build(
        _: &FSSTViewEncoding,
        dtype: &DType,
        len: usize,
        _: &EmptyMetadata,
        buffers: &[ByteBuffer],
        children: &dyn ArrayChildren,
    ) -> VortexResult<FSSTViewArray> {
        // If the DType is nullable, we need to validate the validity information
        vortex_ensure!(
            dtype.is_utf8() || dtype.is_binary(),
            "FSSTViewArray can only be built for utf8 or binary data type, not {dtype}"
        );

        // First buffer are the views.
        let views = Buffer::<View>::from_byte_buffer(buffers[0].clone());

        vortex_ensure!(
            views.len() == len,
            "FSSTViewArray: views expected to have length {len}, was {}",
            views.len()
        );

        // Second buffer are the symbol table
        let symbols = Buffer::<Symbol>::from_byte_buffer(buffers[1].clone());
        // Third buffer: symbol lengths
        let symbol_lengths = buffers[2].clone();

        // Fourth buffer: compressed strings
        let fsst_buffer = buffers[3].clone();

        vortex_ensure!(
            children.len() >= 2,
            "FSSTViewArray: must have 2 or more children"
        );
        let uncompressed_offsets = children.get(0, PType::U32.into(), len + 1)?;
        let compressed_offsets = children.get(1, PType::U32.into(), len + 1)?;

        let has_validity_array = children.len() == 3;

        let validity = match (has_validity_array, dtype.nullability()) {
            // No validity array, nullable => AllValid
            (false, Nullability::Nullable) => Validity::AllValid,
            (false, Nullability::NonNullable) => Validity::NonNullable,
            (true, Nullability::Nullable) => {
                let validity_array =
                    children.get(2, &DType::Binary(Nullability::NonNullable), len)?;
                Validity::Array(validity_array)
            }
            // All other combinations invalid
            (has, nullability) => vortex_bail!(
                "FSSTViewVTable: build: invalid combination: has_validity_array={has} nullability={}",
                nullability.verbose_display()
            ),
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
            views.push(View::new_inlined(&view.as_inlined().data));

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
        unsafe { compressor.compress_into(&buffer[start..end], &mut reuse) };

        let length =
            u32::try_from(reuse.len()).vortex_expect("FSST compressed strings cannot exceed 8MB");
        codes.extend_from_slice(reuse.as_slice());

        let mut prefix = [0u8; 8];
        prefix.copy_from_slice(&reuse[..8]);

        views.push(View {
            outline: OutlinedStr {
                len: length,
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
            .push(uncompressed_offsets.last().copied().unwrap_or(0) + (end - start) as u32);
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
