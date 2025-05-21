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

pub(crate) fn new_exporter(
    array: &RunEndArray,
    cache: &mut ConversionCache,
) -> VortexResult<Box<dyn ArrayExporter>> {
    let ends = array.ends().to_primitive()?;
    let values = array.values().clone();
    let values_exporter = create_exporter(array.values(), cache)?;
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
    ) -> VortexResult<()> {
        let ends_slice = self.ends.as_slice::<E>();

        // Adjust offset to account for the run-end offset.
        let mut offset = E::from_usize(self.run_end_offset + offset)
            .vortex_expect("RunEndExporter::export: offset is not a valid value");
        // Compute the final end offset.
        let end_offset = offset + E::from_usize(len).vortex_expect("len is not end type");

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
            // NOTE(ngates): would be great if we could just export and set type == CONSTANT
            // self.values_exporter.export(start_run_idx, 1, vector, cache);
            let constant = self.values.scalar_at(start_run_idx)?;
            let value = constant.try_to_duckdb_scalar()?;
            vector.flat_vector().assign_to_constant(&value);
            return Ok(());
        }

        // Build up a selection vector
        let mut sel_vec = SelectionVector::new(len as _);
        let mut sel_vec_slice = sel_vec.as_data_slice();

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
        let values_len = *sel_vec.as_data_slice().last().vortex_expect("non-empty") + 1;

        // Export the run-end values into the vector, and then turn it into a dictionary vector.
        self.values_exporter
            .export(start_run_idx, values_len as usize, vector)?;
        vector.flat_vector().slice(values_len as u64, sel_vec);

        Ok(())
    }
}
