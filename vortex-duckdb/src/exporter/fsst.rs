// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ffi::c_char;
use std::mem::align_of;
use std::mem::size_of;

use num_traits::AsPrimitive;
use vortex::array::Array;
use vortex::array::ExecutionCtx;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::arrays::varbin::VarBinArrayExt;
use vortex::array::match_each_integer_ptype;
use vortex::buffer::ByteBuffer;
use vortex::encodings::fsst::FSST;
use vortex::encodings::fsst::FSSTArrayExt;
use vortex::error::VortexResult;

use crate::cpp;
use crate::duckdb::LogicalType;
use crate::duckdb::VectorBuffer;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::all_invalid;
use crate::exporter::validity;

struct FSSTExporter {
    symbols: Vec<u64>,
    symbol_lengths: Vec<u8>,
    offsets: PrimitiveArray,
    bytes: ByteBuffer,
    max_uncompressed_length: usize,
    bytes_buffer: VectorBuffer,
}

pub(crate) fn new_exporter(
    array: Array<FSST>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let len = array.len();
    let codes: VarBinArray = array.codes().clone();
    let validity_mask = codes.varbin_validity_mask();
    if validity_mask.all_false() {
        let logical_type: LogicalType = array.dtype().try_into()?;
        return Ok(all_invalid::new_exporter(len, &logical_type));
    }

    let offsets = codes.offsets().clone().execute::<PrimitiveArray>(ctx)?;
    let uncompressed_lengths = array
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let max_uncompressed_length = match_each_integer_ptype!(uncompressed_lengths.ptype(), |P| {
        uncompressed_lengths
            .as_slice::<P>()
            .iter()
            .map(|len| {
                let len: usize = (*len).as_();
                len
            })
            .max()
            .unwrap_or(0)
    });

    let symbols = array
        .symbols()
        .as_slice()
        .iter()
        .map(|symbol| symbol.to_u64())
        .collect();
    let symbol_lengths = array.symbol_lengths().as_slice().to_vec();
    let bytes = codes.bytes().clone();
    let bytes_buffer = VectorBuffer::new(codes.bytes_handle().clone());

    Ok(validity::new_exporter(
        validity_mask,
        Box::new(FSSTExporter {
            symbols,
            symbol_lengths,
            offsets,
            bytes,
            max_uncompressed_length,
            bytes_buffer,
        }),
    ))
}

impl ColumnExporter for FSSTExporter {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let out = unsafe { vector.as_slice_mut::<PtrString>(len) };
        let bytes = self.bytes.as_ref();
        match_each_integer_ptype!(self.offsets.ptype(), |O| {
            let offsets = self.offsets.as_slice::<O>();
            for row in 0..len {
                let start: usize = offsets[offset + row].as_();
                let end: usize = offsets[offset + row + 1].as_();
                let value = &bytes[start..end];
                out[row] = PtrString::new(value);
            }
        });

        vector.set_fsst(
            &self.symbols,
            &self.symbol_lengths,
            self.max_uncompressed_length,
            len,
            &self.bytes_buffer,
        );

        Ok(())
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
struct PtrString {
    value: PtrStringValue,
}

impl PtrString {
    fn new(value: &[u8]) -> Self {
        let length = u32::try_from(value.len()).expect("FSST code length must fit in u32");
        if value.len() <= 12 {
            let mut inlined = [0_i8; 12];
            for (dst, src) in inlined.iter_mut().zip(value) {
                *dst = *src as i8;
            }
            Self {
                value: PtrStringValue {
                    inlined: PtrStringInlined { length, inlined },
                },
            }
        } else {
            let mut prefix = [0_i8; 4];
            for (dst, src) in prefix.iter_mut().zip(value.iter().copied()) {
                *dst = src as i8;
            }
            Self {
                value: PtrStringValue {
                    pointer: PtrStringPointer {
                        length,
                        prefix,
                        ptr: value.as_ptr().cast_mut().cast::<c_char>(),
                    },
                },
            }
        }
    }
}

#[derive(Clone, Copy)]
#[repr(C)]
union PtrStringValue {
    pointer: PtrStringPointer,
    inlined: PtrStringInlined,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct PtrStringPointer {
    length: u32,
    prefix: [i8; 4],
    ptr: *mut c_char,
}

#[derive(Clone, Copy)]
#[repr(C)]
struct PtrStringInlined {
    length: u32,
    inlined: [i8; 12],
}

const _: () = {
    assert!(size_of::<PtrString>() == size_of::<cpp::duckdb_string_t>());
    assert!(align_of::<PtrString>() == align_of::<cpp::duckdb_string_t>());
};

#[cfg(test)]
mod tests {
    use num_traits::AsPrimitive;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::VarBinArray;
    use vortex::array::arrays::varbin::VarBinArrayExt;
    use vortex::array::match_each_integer_ptype;
    use vortex::dtype::DType;
    use vortex::dtype::Nullability;
    use vortex::encodings::fsst::fsst_compress;
    use vortex::encodings::fsst::fsst_train_compressor;
    use vortex::error::VortexResult;

    use crate::SESSION;
    use crate::cpp;
    use crate::duckdb::DataChunk;
    use crate::duckdb::ExtractedValue;
    use crate::duckdb::LogicalType;
    use crate::exporter::fsst::new_exporter;

    #[test]
    fn fsst_utf8_exports_as_fsst_vector() -> VortexResult<()> {
        let values = VarBinArray::from_iter(
            [
                Some("hello"),
                None,
                Some("compressed world"),
                Some("duckdb"),
            ],
            DType::Utf8(Nullability::Nullable),
        );
        let compressor = fsst_train_compressor(&values);
        let array = fsst_compress(
            &values,
            values.len(),
            &DType::Utf8(Nullability::Nullable),
            &compressor,
        );

        let mut chunk = DataChunk::new([LogicalType::varchar()]);
        let mut ctx = SESSION.create_execution_ctx();
        new_exporter(array, &mut ctx)?.export(0, 4, chunk.get_vector_mut(0), &mut ctx)?;
        chunk.set_len(4);

        assert!(chunk.get_vector(0).is_fsst());

        chunk.get_vector(0).flatten(4);
        assert_eq!(
            format!("{}", String::try_from(&*chunk)?),
            r#"Chunk - [1 Columns]
- FLAT VARCHAR: 4 = [ hello, NULL, compressed world, duckdb]
"#
        );
        Ok(())
    }

    #[test]
    fn fsst_blob_exports_as_fsst_vector() -> VortexResult<()> {
        let values = VarBinArray::from_iter(
            [
                Some(&b"\x00\x01foo"[..]),
                None,
                Some(&b"longer compressed blob"[..]),
            ],
            DType::Binary(Nullability::Nullable),
        );
        let compressor = fsst_train_compressor(&values);
        let array = fsst_compress(
            &values,
            values.len(),
            &DType::Binary(Nullability::Nullable),
            &compressor,
        );

        let mut chunk = DataChunk::new([LogicalType::new(cpp::duckdb_type::DUCKDB_TYPE_BLOB)]);
        let mut ctx = SESSION.create_execution_ctx();
        new_exporter(array, &mut ctx)?.export(0, 3, chunk.get_vector_mut(0), &mut ctx)?;
        chunk.set_len(3);

        assert!(chunk.get_vector(0).is_fsst());

        let values = (0..3)
            .map(|idx| {
                chunk
                    .get_vector(0)
                    .get_value(idx, 3)
                    .expect("value must exist")
                    .extract()
            })
            .collect::<Vec<_>>();

        assert!(
            matches!(&values[0], ExtractedValue::Blob(bytes) if bytes.as_ref() == b"\x00\x01foo")
        );
        assert!(matches!(&values[1], ExtractedValue::Null));
        assert!(
            matches!(&values[2], ExtractedValue::Blob(bytes) if bytes.as_ref() == b"longer compressed blob")
        );
        Ok(())
    }

    #[test]
    fn fsst_utf8_exports_non_inline_codes() -> VortexResult<()> {
        let values = VarBinArray::from_iter(
            [
                Some("abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
                Some("ZYXWVUTSRQPONMLKJIHGFEDCBA9876543210abcdefghijklmnopqrstuvwxyz"),
            ],
            DType::Utf8(Nullability::Nullable),
        );
        let compressor = fsst_train_compressor(&values);
        let array = fsst_compress(
            &values,
            values.len(),
            &DType::Utf8(Nullability::Nullable),
            &compressor,
        );

        let mut ctx = SESSION.create_execution_ctx();
        let offsets = array
            .codes()
            .offsets()
            .clone()
            .execute::<PrimitiveArray>(&mut ctx)?;
        let has_non_inline_code = match_each_integer_ptype!(offsets.ptype(), |O| {
            offsets.as_slice::<O>().windows(2).any(|window| {
                let start: usize = window[0].as_();
                let end: usize = window[1].as_();
                end - start > 12
            })
        });
        assert!(
            has_non_inline_code,
            "test data must exercise the pointer-backed string path"
        );

        let mut chunk = DataChunk::new([LogicalType::varchar()]);
        new_exporter(array, &mut ctx)?.export(0, 2, chunk.get_vector_mut(0), &mut ctx)?;
        chunk.set_len(2);

        assert!(chunk.get_vector(0).is_fsst());

        chunk.get_vector(0).flatten(2);
        assert_eq!(
            format!("{}", String::try_from(&*chunk)?),
            r#"Chunk - [1 Columns]
- FLAT VARCHAR: 2 = [ abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ, ZYXWVUTSRQPONMLKJIHGFEDCBA9876543210abcdefghijklmnopqrstuvwxyz]
"#
        );
        Ok(())
    }
}
