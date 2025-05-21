use std::marker::PhantomData;

use duckdb::core::SelectionVector;
use duckdb::vtab::arrow::WritableVector;
use num_traits::{FromPrimitive, ToPrimitive};
use vortex_array::arrays::PrimitiveArray;
use vortex_array::search_sorted::{SearchSorted, SearchSortedSide};
use vortex_array::{ArrayRef, ToCanonical};
use vortex_dtype::{NativePType, match_each_integer_ptype};
use vortex_error::{VortexExpect, VortexResult};
use vortex_mask::Mask;
use vortex_runend::RunEndArray;

use crate::exporter::create_exporter;
use crate::{ArrayExporter, ConversionCache, ToDuckDBScalar};

/// We export run-end arrays to a DuckDB dictionary vector, using a selection vector to
/// repeat the values in the run-end array.
#[allow(dead_code)]
pub(crate) struct RunEndExporter<E: NativePType> {
    ends: PrimitiveArray,
    ends_type: PhantomData<E>,
    values: ArrayRef,
    values_exporter: Box<dyn ArrayExporter>,
    validity: Mask,
    run_end_offset: usize,
}

pub(crate) fn new_exporter(array: &RunEndArray) -> VortexResult<Box<dyn ArrayExporter>> {
    let ends = array.ends().to_primitive()?;
    let values = array.values().clone();
    let values_exporter = create_exporter(array.values())?;
    let validity = array.validity_mask()?;

    match_each_integer_ptype!(ends.ptype(), |$E| {
        Ok(Box::new(RunEndExporter {
            ends,
            ends_type: PhantomData::<$E>,
            values,
            values_exporter,
            validity,
            run_end_offset: array.offset(),
        }))
    })
}

impl<E: NativePType + Ord + FromPrimitive + ToPrimitive> ArrayExporter for RunEndExporter<E> {
    fn export(
        &self,
        offset: usize,
        len: usize,
        vector: &mut dyn WritableVector,
        cache: &mut ConversionCache,
    ) -> VortexResult<()> {
        let ends_slice = self.ends.as_slice::<E>();

        // Adjust offset to account for the run-end offset.
        let mut offset = E::from_usize(self.run_end_offset + offset)
            .vortex_expect("RunEndExporter::export: offset is not a valid value");

        // Find the run that contains the start offset.
        let start_run_idx = ends_slice
            .search_sorted(&offset, SearchSortedSide::Right)
            .to_ends_index(ends_slice.len());

        // Find the final run in case we can short-circuit and return a constant vector.
        let end_run_idx = ends_slice
            .search_sorted(
                &offset.add(E::from_usize(len).vortex_expect("len out of bounds")),
                SearchSortedSide::Right,
            )
            .to_ends_index(ends_slice.len());

        if start_run_idx == end_run_idx {
            // TODO(ngates): would be great if we could just export and set type == CONSTANT
            // self.values_exporter.export(start_run_idx, 1, vector, cache);
            let constant = self.values.scalar_at(start_run_idx)?;
            let value = constant.try_to_duckdb_scalar()?;
            vector.flat_vector().assign_to_constant(&value);
            return Ok(());
        }

        // Build up a selection vector
        let mut sel_vec = SelectionVector::new(len as _);
        let sel_vec_slice = sel_vec.as_data_slice();

        // The current run to index.
        let mut run_idx = start_run_idx;
        // The start idx in the values array.
        let values_start = run_idx;
        // The number of values we have selected thus far.
        let mut selected = 0;
        while selected < len {
            let next_offset = ends_slice[run_idx];
            let run_len = (next_offset - offset)
                .to_usize()
                .vortex_expect("run_len is usize")
                .min(len - selected);

            // Push the runs into the selection vector.
            sel_vec_slice[selected..selected + run_len]
                .fill(u32::try_from(run_idx - values_start).vortex_expect("sel_idx is u32"));

            run_idx += 1;
            selected += run_len;
            offset = next_offset;
        }
        let values_stop = run_idx;
        let values_len = values_stop - values_start;

        // Export the run-end values into the vector, and then turn it into a dictionary vector.
        self.values_exporter
            .export(values_start, values_len, vector, cache)?;
        vector.flat_vector().slice(values_len as u64, sel_vec);

        Ok(())
    }
}
