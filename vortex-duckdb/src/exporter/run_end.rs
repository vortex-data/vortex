// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;

use vortex::array::ArrayRef;
use vortex::array::ExecutionCtx;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::match_each_integer_ptype;
use vortex::array::search_sorted::SearchSorted;
use vortex::array::search_sorted::SearchSortedSide;
use vortex::dtype::IntegerPType;
use vortex::encodings::runend::RunEndArray;
use vortex::encodings::runend::RunEndArrayParts;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;

use crate::convert::ToDuckDBScalar;
use crate::duckdb::SelectionVector;
use crate::duckdb::VectorRef;
use crate::exporter::ColumnExporter;
use crate::exporter::cache::ConversionCache;
use crate::exporter::new_array_exporter;

/// We export run-end arrays to a DuckDB dictionary vector, using a selection vector to
/// repeat the values in the run-end array.
struct RunEndExporter<E: IntegerPType> {
    ends: PrimitiveArray,
    ends_type: PhantomData<E>,
    values: ArrayRef,
    values_exporter: Box<dyn ColumnExporter>,
    run_end_offset: usize,
}

pub(crate) fn new_exporter(
    array: RunEndArray,
    cache: &ConversionCache,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Box<dyn ColumnExporter>> {
    let offset = array.offset();
    let RunEndArrayParts { ends, values } = array.into_data().into_parts();
    let ends = ends.execute::<PrimitiveArray>(ctx)?;
    let values_exporter = new_array_exporter(values.clone(), cache, ctx)?;

    match_each_integer_ptype!(ends.ptype(), |E| {
        Ok(Box::new(RunEndExporter {
            ends,
            ends_type: PhantomData::<E>,
            values,
            values_exporter,
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
            .search_sorted(
                &offset.add(E::from_usize(len).vortex_expect("len out of bounds")),
                SearchSortedSide::Right,
            )?
            .to_ends_index(ends_slice.len());

        if start_run_idx == end_run_idx {
            // NOTE(ngates): would be great if we could just export and set type == CONSTANT
            // self.values_exporter.export(start_run_idx, 1, vector, cache);
            let constant = self.values.scalar_at(start_run_idx)?;
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

            // Push the runs into the selection vector.
            sel_vec_slice[..run_len].fill(u32::try_from(run_idx).vortex_expect("sel_idx is u32"));
            sel_vec_slice = &mut sel_vec_slice[run_len..];

            offset = next_end;
        }
        assert!(
            sel_vec_slice.is_empty(),
            "Selection vector not completely filled"
        );

        // The values in the selection vector are the run indices, so we can find the number of
        // values we referenced by looking at the last index of the selection vector.
        let values_len = *unsafe { sel_vec.as_slice_mut(len) }
            .last()
            .vortex_expect("non-empty")
            + 1;

        // Export the run-end values into the vector, and then turn it into a dictionary vector.
        self.values_exporter
            .export(start_run_idx, values_len as usize, vector, ctx)?;
        vector.dictionary(vector, values_len as usize, &sel_vec, len as _);

        Ok(())
    }
}
