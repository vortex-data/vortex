// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Batch-sized run selections over a cached DuckDB values dictionary.
//!
//! [`DictionaryParts`] lets the chunked exporter join run-end and dictionary chunks that share
//! values without flattening them. Codes are expanded only for the requested batch. Routing through
//! [`RunEndArray::to_dict`] would make the generic dictionary exporter materialize all codes at setup.

use std::marker::PhantomData;

use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::match_each_integer_ptype;
use vortex::dtype::IntegerPType;
use vortex::encodings::runend::RunEndArray;
use vortex::encodings::runend::RunEndArrayExt;
use vortex::encodings::runend::RunEndArraySlotsExt;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;

use crate::convert::ToDuckDBScalar;
use crate::duckdb::ReusableDict;
use crate::duckdb::SelectionVector;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::DictionaryParts;
use crate::exporter::cache::ConversionCache;
use crate::exporter::cached_values_dict;
use crate::exporter::canonical;

/// Keeps the values dictionary cached while generating selections for each requested batch.
struct RunEndExporter<E: IntegerPType> {
    ends: PrimitiveArray,
    ends_type: PhantomData<E>,
    values: ArrayRef,
    values_dict: ReusableDict,
    /// Added to request offsets because sliced run ends retain their source coordinates.
    run_end_offset: usize,
}

pub(crate) fn new_exporter_with_flatten(
    array: RunEndArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
    flatten: bool,
) -> VortexResult<Box<dyn ColumnExporter>> {
    if array.is_empty() {
        return canonical::new_exporter(Canonical::empty(array.dtype()).into_array(), cache, ctx);
    }

    // Canonicalize directly to avoid building a dictionary that DuckDB would immediately flatten.
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

impl<E: IntegerPType> RunEndExporter<E> {
    /// Finds the run containing an absolute row offset. Run ends are exclusive.
    ///
    /// The offset must refer to an existing row, rather than the end of an empty request.
    fn first_run(&self, offset: usize) -> usize {
        let offset =
            E::from_usize(offset).vortex_expect("the request offset fits the run-end type");
        self.ends
            .as_slice::<E>()
            .partition_point(|&end| end <= offset)
    }

    /// Fills a non-empty request with indices into the original values child.
    ///
    /// `offset` is absolute, and `first_run` must be the index returned by [`Self::first_run`].
    fn write_run_codes(&self, offset: usize, first_run: usize, codes: &mut [u32]) {
        let mut position =
            E::from_usize(offset).vortex_expect("the request offset fits the run-end type");
        let end_offset = position
            + E::from_usize(codes.len()).vortex_expect("the request length fits the run-end type");
        let mut remaining = codes;

        for (index, &end) in self.ends.as_slice::<E>()[first_run..].iter().enumerate() {
            let next_end = end.min(end_offset);
            let run_len = (next_end - position)
                .to_usize()
                .vortex_expect("run length fits usize");
            let code = u32::try_from(first_run + index).vortex_expect("run index exceeds u32");
            remaining[..run_len].fill(code);
            remaining = &mut remaining[run_len..];
            if remaining.is_empty() {
                return;
            }
            position = next_end;
        }

        assert!(
            remaining.is_empty(),
            "run ends must cover the requested range"
        );
    }
}

impl<E: IntegerPType> DictionaryParts for RunEndExporter<E> {
    fn values(&self) -> &ReusableDict {
        &self.values_dict
    }

    fn values_key(&self) -> usize {
        self.values.addr()
    }

    fn write_codes(&self, offset: usize, codes: &mut [u32]) {
        if codes.is_empty() {
            return;
        }

        let offset = self.run_end_offset + offset;
        self.write_run_codes(offset, self.first_run(offset), codes);
    }
}

impl<E: IntegerPType> ColumnExporter for RunEndExporter<E> {
    fn as_dictionary(&self) -> Option<&dyn DictionaryParts> {
        Some(self)
    }

    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut VectorRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        if len == 0 {
            return Ok(());
        }

        let offset = self.run_end_offset + offset;
        let first_run = self.first_run(offset);
        let end =
            E::from_usize(offset + len).vortex_expect("the request end fits the run-end type");

        // The request end is exclusive, so equality still leaves every row inside this run.
        if end <= self.ends.as_slice::<E>()[first_run] {
            let constant = self.values.execute_scalar(first_run, ctx)?;
            let value = constant.try_to_duckdb_scalar()?;
            vector.reference_value(&value);
            return Ok(());
        }

        let mut selection = SelectionVector::with_capacity(len);
        // SAFETY: the selection has capacity for `len` entries. The validated run ends cover the
        // requested range, and write_run_codes fills every entry before DuckDB reads the selection.
        let codes = unsafe { selection.as_slice_mut(len) };
        self.write_run_codes(offset, first_run, codes);
        vector.reuse_dictionary(&self.values_dict, &selection, len);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::array::ArrayRef;
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::ChunkedArray;
    use vortex::array::arrays::DictArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::StructArray;
    use vortex::buffer::Buffer;
    use vortex::buffer::buffer;
    use vortex::encodings::runend::RunEnd;
    use vortex::error::VortexResult;

    use crate::SESSION;
    use crate::duckdb::DataChunk;
    use crate::duckdb::LogicalType;
    use crate::exporter::ArrayExporter;
    use crate::exporter::ConversionCache;
    use crate::exporter::new_array_exporter;

    #[track_caller]
    fn check_exports(array: ArrayRef, requests: &[(usize, usize, &str)]) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let exporter = new_array_exporter(array, &ConversionCache::default(), &mut ctx)?;
        let mut chunk = DataChunk::new([LogicalType::int32()]);

        for &(offset, len, expected) in requests {
            chunk.reset();
            exporter.export(offset, len, chunk.get_vector_mut(0), &mut ctx)?;
            chunk.set_len(len);
            assert_eq!(
                String::try_from(&*chunk)?,
                format!("Chunk - [1 Columns]\n{expected}\n")
            );
        }

        Ok(())
    }

    #[test]
    fn test_one_chunk_null() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let source = vec![Some(0u32), Some(1), None, Some(3), None];
        let array = PrimitiveArray::from_option_iter(source);
        let array = RunEnd::encode(array.into_array(), &mut ctx)?;

        check_exports(
            array.into_array(),
            &[(0, 5, "- DICTIONARY INTEGER: 5 = [ 0, 1, NULL, 3, NULL]")],
        )
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

        assert!(exporter.export(&mut chunk, None)?);
        assert_eq!(
            String::try_from(&*chunk)?,
            r#"Chunk - [1 Columns]
- DICTIONARY INTEGER: 2 = [ 10, 20]
"#
        );

        assert!(!exporter.export(&mut chunk, None)?);
        Ok(())
    }

    #[test]
    fn sliced_batches_keep_run_boundaries() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let array = RunEnd::try_new_offset_length(
            buffer![3u32, 5, 8].into_array(),
            PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array(),
            2,
            5,
            &mut ctx,
        )?;

        check_exports(
            array.into_array(),
            &[
                (0, 1, "- CONSTANT INTEGER: 1 = [ 10]"), // Exact first run end.
                (1, 2, "- CONSTANT INTEGER: 2 = [ NULL]"), // Exact null run.
                (2, 3, "- DICTIONARY INTEGER: 3 = [ NULL, 30, 30]"), // Clipped batch.
                (5, 0, "- FLAT INTEGER: 0 = [ ]"),       // Empty request at the end.
            ],
        )
    }

    #[test]
    fn empty_arrays_do_not_cache_values() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = buffer![10i32].into_array();
        let retained =
            RunEnd::try_new_offset_length(buffer![3u32].into_array(), values, 3, 0, &mut ctx)?;
        let empty = RunEnd::try_new(
            Buffer::<u32>::empty().into_array(),
            Buffer::<i32>::empty().into_array(),
            &mut ctx,
        )?;

        for array in [empty, retained] {
            let cache = ConversionCache::default();
            let exporter = new_array_exporter(array.into_array(), &cache, &mut ctx)?;
            let mut chunk = DataChunk::new([LogicalType::int32()]);
            exporter.export(0, 0, chunk.get_vector_mut(0), &mut ctx)?;
            assert!(cache.dict_cache.is_empty());
        }

        Ok(())
    }

    #[rstest]
    #[case::runend(false, true)]
    #[case::dictionary(true, true)]
    #[case::different_values(false, false)]
    fn chunk_boundaries(#[case] dictionary: bool, #[case] shared: bool) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array();
        let first = RunEnd::try_new_offset_length(
            buffer![2u32, 4, 6].into_array(),
            values.clone(),
            2,
            4,
            &mut ctx,
        )?
        .into_array();
        let next_values = if shared {
            values
        } else {
            PrimitiveArray::from_option_iter([Some(10i32), None, Some(30)]).into_array()
        };
        let second = if dictionary {
            DictArray::try_new(buffer![0u32, 1, 1, 2].into_array(), next_values)?.into_array()
        } else {
            RunEnd::try_new(buffer![1u32, 3, 4].into_array(), next_values, &mut ctx)?.into_array()
        };
        let dtype = first.dtype().clone();
        let array = ChunkedArray::try_new(vec![first, second], dtype)?.into_array();
        let crossing = if shared {
            "- DICTIONARY INTEGER: 6 = [ NULL, 30, 30, 10, NULL, NULL]"
        } else {
            "- FLAT INTEGER: 6 = [ NULL, 30, 30, 10, NULL, NULL]"
        };

        check_exports(array, &[(1, 6, crossing)])
    }
}
