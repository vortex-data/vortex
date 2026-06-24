// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::match_each_integer_ptype;
use vortex::array::search_sorted::SearchSorted;
use vortex::array::search_sorted::SearchSortedSide;
use vortex::dtype::IntegerPType;
use vortex::encodings::runend::RunEndArray;
use vortex::encodings::runend::RunEndArrayExt;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;

use crate::convert::ToDuckDBScalar;
use crate::duckdb::ReusableDict;
use crate::duckdb::SelectionVector;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::cache::ConversionCache;
use crate::exporter::cached_values_dict;
use crate::exporter::canonical;

/// We export run-end arrays to a DuckDB dictionary vector. Values are exported
/// into a ReusableDict with SelectionVector applied in export().
struct RunEndExporter<E: IntegerPType> {
    ends: PrimitiveArray,
    ends_type: PhantomData<E>,
    values: ArrayRef,
    values_dict: ReusableDict,
    run_end_offset: usize,
}

pub(crate) fn new_exporter_with_flatten(
    array: RunEndArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
    flatten: bool,
) -> VortexResult<Box<dyn ColumnExporter>> {
    // Our canonicalization is faster than creating a dictionary vector and
    // letting duckdb flatten it for us.
    if flatten {
        return canonical::new_exporter(array.into_array(), cache, ctx);
    }

    let offset = array.offset();
    let ends = array.ends().clone();
    let values = array.values().clone();
    let ends = ends.execute::<PrimitiveArray>(ctx)?;
    let values_dict = cached_values_dict(values.clone(), cache, ctx)?;

    match_each_integer_ptype!(ends.ptype(), |E| {
        Ok(Box::new(RunEndExporter {
            ends,
            ends_type: PhantomData::<E>,
            values,
            values_dict,
            run_end_offset: offset,
        }))
    })
}

impl<E: IntegerPType> ColumnExporter for RunEndExporter<E> {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let ends_slice = self.ends.as_slice::<E>();

        // Adjust offset to account for the run-end offset.
        let mut offset = E::from_usize(self.run_end_offset + offset)
            .vortex_expect("RunEndExporter::export: offset is not a valid value");
        // Compute the final end offset.
        let end_offset = offset + E::from_usize(len).vortex_expect("len is not end type");

        // Find the run that contains the start offset.
        let start_run_idx = ends_slice
            .search_sorted(&offset, SearchSortedSide::Right)?
            .to_ends_index(ends_slice.len());

        // Find the final run in case we can short-circuit and return a constant vector.
        let end_run_idx = ends_slice
            .search_sorted(&end_offset, SearchSortedSide::Right)?
            .to_ends_index(ends_slice.len());

        if start_run_idx == end_run_idx {
            // NOTE(ngates): would be great if we could just export and set type == CONSTANT
            // self.values_exporter.export(start_run_idx, 1, vector, cache);
            let constant = self.values.execute_scalar(start_run_idx, ctx)?;
            let value = constant.try_to_duckdb_scalar()?;
            vector.reference_value(&value);
            return Ok(());
        }

        // Build up a selection vector
        let mut sel_vec = SelectionVector::with_capacity(len);
        let mut sel_vec_slice = unsafe { sel_vec.as_slice_mut(len) };

        for (run_idx, &next_end) in ends_slice[start_run_idx..=end_run_idx].iter().enumerate() {
            let next_end = next_end.min(end_offset);
            let run_len = (next_end - offset)
                .to_usize()
                .vortex_expect("run_len is usize");

            let global_run_idx =
                u32::try_from(start_run_idx + run_idx).vortex_expect("run index exceeds u32");
            sel_vec_slice[..run_len].fill(global_run_idx);
            sel_vec_slice = &mut sel_vec_slice[run_len..];

            offset = next_end;
        }
        debug_assert!(sel_vec_slice.is_empty());

        vector.reuse_dictionary(&self.values_dict, &sel_vec);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::ChunkedArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::buffer::buffer;
    use vortex::encodings::runend::RunEnd;
    use vortex::error::VortexResult;

    use crate::SESSION;
    use crate::cpp::duckdb_type::DUCKDB_TYPE_INTEGER;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;
    use crate::exporter::ArrayExporter;
    use crate::exporter::ConversionCache;
    use crate::exporter::new_array_exporter;

    #[test]
    fn test_one_chunk_null() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source = vec![Some(0u32), Some(1), None, Some(3), None];
        let array = PrimitiveArray::from_option_iter(source);
        let array = RunEnd::encode(array.into_array(), &mut ctx)?;

        let mut chunk = DataChunk::new([LogicalType::new(DUCKDB_TYPE_INTEGER)]);
        new_array_exporter(array.into_array(), &ConversionCache::default(), &mut ctx)?.export(
            0,
            5,
            chunk.get_vector_mut(0),
            &mut ctx,
        )?;
        chunk.set_len(5);
        let chunk_str = String::try_from(&*chunk)?;
        assert_eq!(
            chunk_str,
            r#"Chunk - [1 Columns]
- DICTIONARY INTEGER: 5 = [ 0, 1, NULL, 3, NULL]
"#
        );
        Ok(())
    }

    #[test]
    fn run_end_with_chunked_values_exports_across_value_chunks() -> VortexResult<()> {
        let values0 = PrimitiveArray::from_iter([10i32]).into_array();
        let dtype = values0.dtype().clone();
        let values1 = PrimitiveArray::from_iter([20i32]).into_array();
        let values = ChunkedArray::try_new(vec![values0, values1], dtype)?.into_array();
        let mut ctx = SESSION.create_execution_ctx();
        let field = RunEnd::try_new(buffer![1u32, 2].into_array(), values, &mut ctx)?.into_array();
        let array = StructArray::from_fields(&[("field", field)])?;
        let mut exporter = ArrayExporter::try_new(&array, &ConversionCache::default(), ctx)?;
        let mut chunk = DataChunk::new([LogicalType::int32()]);

        assert!(exporter.export(&mut chunk, None, None)?);
        assert_eq!(
            format!("{}", String::try_from(&*chunk)?),
            r#"Chunk - [1 Columns]
- DICTIONARY INTEGER: 2 = [ 10, 20]
"#
        );

        assert!(!exporter.export(&mut chunk, None, None)?);
        Ok(())
    }
}
