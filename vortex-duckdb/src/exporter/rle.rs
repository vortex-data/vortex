// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use num_traits::AsPrimitive;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::match_each_unsigned_integer_ptype;
use vortex::array::validity::Validity;
use vortex::dtype::IntegerPType;
use vortex::encodings::fastlanes::FL_CHUNK_SIZE;
use vortex::encodings::fastlanes::RLEArray;
use vortex::encodings::fastlanes::RLEArrayExt;
use vortex::error::VortexResult;

use crate::duckdb::ReusableDict;
use crate::duckdb::SelectionVector;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::all_invalid;
use crate::exporter::cache::ConversionCache;
use crate::exporter::cached_values_dict;
use crate::exporter::canonical;

struct RLEExporter<I: IntegerPType, O: IntegerPType> {
    values: ReusableDict,
    indices: PrimitiveArray,
    values_idx_offsets: PrimitiveArray,
    /// Offset relative to the first chunk
    offset: usize,
    indices_type: PhantomData<I>,
    values_idx_offsets_type: PhantomData<O>,
}

pub(crate) fn new_exporter_with_flatten(
    array: RLEArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
    flatten: bool,
) -> VortexResult<Box<dyn ColumnExporter>> {
    if flatten || array.is_empty() {
        return canonical::new_exporter(array.into_array(), cache, ctx);
    }
    // DuckDB dictionary can't carry validity on codes.
    // Don't execute the validity mask, if there's a chance of NULL,
    // canonicalize
    match array.indices().validity()? {
        Validity::AllInvalid => return Ok(all_invalid::new_exporter()),
        Validity::Array(_) => return canonical::new_exporter(array.into_array(), cache, ctx),
        _ => {}
    }

    let indices = array.indices().clone().execute::<PrimitiveArray>(ctx)?;
    let values = array.values().clone();
    let values_idx_offsets = array
        .values_idx_offsets()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;

    let values = cached_values_dict(values, cache, ctx)?;
    match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
        match_each_unsigned_integer_ptype!(values_idx_offsets.ptype(), |O| {
            Ok(Box::new(RLEExporter {
                values,
                indices,
                values_idx_offsets,
                offset: array.offset(),
                indices_type: PhantomData::<I>,
                values_idx_offsets_type: PhantomData::<O>,
            }))
        })
    })
}

impl<I, O> ColumnExporter for RLEExporter<I, O>
where
    I: IntegerPType + AsPrimitive<u32>,
    O: IntegerPType + AsPrimitive<u32>,
{
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let mut selection_vec = SelectionVector::with_capacity(len);
        let mut selection = unsafe { selection_vec.as_slice_mut(len) };

        let indices = self.indices.as_slice::<I>();
        let values_idx_offsets = self.values_idx_offsets.as_slice::<O>();

        let mut pos = self.offset + offset;
        let end = pos + len;

        let first_idx_offset = values_idx_offsets[0];
        while pos < end {
            let chunk_idx = pos / FL_CHUNK_SIZE;
            let base: u32 = (values_idx_offsets[chunk_idx] - first_idx_offset).as_();
            let take = ((chunk_idx + 1) * FL_CHUNK_SIZE).min(end) - pos;

            for (dst, idx) in selection[..take].iter_mut().zip(&indices[pos..pos + take]) {
                let idx: u32 = idx.as_();
                *dst = base + idx;
            }

            selection = &mut selection[take..];
            pos += take;
        }

        vector.reuse_dictionary(&self.values, &selection_vec);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::ArrayRef;
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::encodings::fastlanes::RLEArray;
    use vortex::encodings::fastlanes::RLEData;
    use vortex::error::VortexResult;

    use crate::SESSION;
    use crate::cpp::duckdb_type::DUCKDB_TYPE_INTEGER;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;
    use crate::exporter::ConversionCache;
    use crate::exporter::new_array_exporter;

    fn encode_rle(values: Vec<i32>) -> VortexResult<RLEArray> {
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_iter(values);
        RLEData::encode(primitive.as_view(), &mut ctx)
    }

    fn export_flat(array: ArrayRef, len: usize) -> VortexResult<Vec<i32>> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut chunk = DataChunk::new([LogicalType::new(DUCKDB_TYPE_INTEGER)]);
        new_array_exporter(array, &ConversionCache::default(), &mut ctx)?.export(
            0,
            len,
            chunk.get_vector_mut(0),
            &mut ctx,
        )?;
        chunk.set_len(len);
        let vector = chunk.get_vector(0);
        vector.flatten(len as u64);
        Ok(vector.as_slice_with_len::<i32>(len).to_vec())
    }

    #[test]
    fn test_roundtrip_two_chunks() -> VortexResult<()> {
        let expected: Vec<i32> = (0i32..2048).map(|i| i / 100).collect();
        let rle = encode_rle(expected.clone())?;
        let exported = export_flat(rle.into_array(), 2048)?;
        assert_eq!(exported, expected);
        Ok(())
    }

    #[test]
    fn test_roundtrip_boundary() -> VortexResult<()> {
        let source: Vec<i32> = (0i32..2048).map(|i| i / 100).collect();
        let rle = encode_rle(source.clone())?;
        let sliced = rle.into_array().slice(500..1700)?;
        let exported = export_flat(sliced, 1200)?;
        assert_eq!(exported, source[500..1700]);
        Ok(())
    }

    #[test]
    fn test_roundtrip_slice() -> VortexResult<()> {
        let source: Vec<i32> = (0i32..3072).map(|i| i / 100).collect();
        let rle = encode_rle(source.clone())?;
        let sliced = rle.into_array().slice(1200..2000)?;
        let exported = export_flat(sliced, 800)?;
        assert_eq!(exported, source[1200..2000]);
        Ok(())
    }

    fn chunk_string(array: ArrayRef, offset: usize, len: usize) -> VortexResult<String> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut chunk = DataChunk::new([LogicalType::new(DUCKDB_TYPE_INTEGER)]);
        new_array_exporter(array, &ConversionCache::default(), &mut ctx)?.export(
            offset,
            len,
            chunk.get_vector_mut(0),
            &mut ctx,
        )?;
        chunk.set_len(len);
        String::try_from(&*chunk)
    }

    fn two_chunk_rle() -> VortexResult<RLEArray> {
        let mut ctx = SESSION.create_execution_ctx();
        let source: Vec<i32> = std::iter::repeat_n(10i32, 1024)
            .chain(std::iter::repeat_n(20, 1024))
            .collect();
        RLEData::encode(PrimitiveArray::from_iter(source).as_view(), &mut ctx)
    }

    #[test]
    fn test_one_chunk() -> VortexResult<()> {
        let rle = two_chunk_rle()?;
        let chunk_str = chunk_string(rle.into_array(), 0, 5)?;
        assert_eq!(
            chunk_str,
            r#"Chunk - [1 Columns]
- DICTIONARY INTEGER: 5 = [ 10, 10, 10, 10, 10]
"#
        );
        Ok(())
    }

    #[test]
    fn test_one_chunk_nulls() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source = vec![Some(0u32), Some(1), None, Some(3), None];
        let rle = RLEData::encode(PrimitiveArray::from_option_iter(source).as_view(), &mut ctx)?;
        let chunk_str = chunk_string(rle.into_array(), 0, 5)?;
        assert_eq!(
            chunk_str,
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 5 = [ 0, 1, NULL, 3, NULL]
"#
        );
        Ok(())
    }

    #[test]
    fn test_chunk_boundary() -> VortexResult<()> {
        let rle = two_chunk_rle()?;
        let chunk_str = chunk_string(rle.into_array(), 1020, 10)?;
        assert_eq!(
            chunk_str,
            r#"Chunk - [1 Columns]
- DICTIONARY INTEGER: 10 = [ 10, 10, 10, 10, 20, 20, 20, 20, 20, 20]
"#
        );
        Ok(())
    }

    #[test]
    fn test_chunk_slice() -> VortexResult<()> {
        let rle = two_chunk_rle()?;
        let sliced = rle.into_array().slice(1500..1510)?;
        let chunk_str = chunk_string(sliced, 0, 10)?;
        assert_eq!(
            chunk_str,
            r#"Chunk - [1 Columns]
- FLAT INTEGER: 10 = [ 20, 20, 20, 20, 20, 20, 20, 20, 20, 20]
"#
        );
        Ok(())
    }

    #[test]
    fn test_roundtrip_with_nulls() -> VortexResult<()> {
        let source: Vec<Option<i32>> = (0i32..1024)
            .map(|i| if i % 7 == 0 { None } else { Some(i / 50) })
            .collect();
        let mut ctx = SESSION.create_execution_ctx();
        let primitive = PrimitiveArray::from_option_iter(source.clone());
        let rle = RLEData::encode(primitive.as_view(), &mut ctx)?;

        let mut chunk = DataChunk::new([LogicalType::new(DUCKDB_TYPE_INTEGER)]);
        new_array_exporter(rle.into_array(), &ConversionCache::default(), &mut ctx)?.export(
            0,
            1024,
            chunk.get_vector_mut(0),
            &mut ctx,
        )?;
        chunk.set_len(1024);

        let vector = chunk.get_vector(0);
        vector.flatten(1024);
        let slice = vector.as_slice_with_len::<i32>(1024);
        for (i, expected) in source.iter().enumerate() {
            if let Some(v) = expected {
                assert!(!vector.row_is_null(i as u64), "row {i} is null");
                assert_eq!(slice[i], *v);
            } else {
                assert!(vector.row_is_null(i as u64), "row {i} not null");
            }
        }
        Ok(())
    }
}
