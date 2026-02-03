// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;
use vortex_mask::MaskIter;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ChunkedArray;
use crate::arrays::ChunkedVTable;
use crate::arrays::FilterArray;
use crate::arrays::FilterVTable;
use crate::arrays::chunked::compute::filter::FILTER_SLICES_SELECTIVITY_THRESHOLD;
use crate::arrays::chunked::compute::filter::filter_indices;
use crate::arrays::chunked::compute::filter::filter_slices;
use crate::kernel::ExecuteParentKernel;
use crate::matchers::Exact;

#[derive(Debug)]
pub(super) struct ChunkedFilterKernel;

impl ExecuteParentKernel<ChunkedVTable> for ChunkedFilterKernel {
    type Parent = Exact<FilterVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::new()
    }

    fn execute_parent(
        &self,
        array: &ChunkedArray,
        parent: &FilterArray,
        _child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let mask = parent.filter_mask();

        // Handle trivial cases
        let mask_values = match mask {
            Mask::AllTrue(_) | Mask::AllFalse(_) => return Ok(None),
            Mask::Values(v) => v,
        };

        // Based on filter selectivity, we take the values between a range of slices, or
        // we take individual indices.
        let chunks = match mask_values.threshold_iter(FILTER_SLICES_SELECTIVITY_THRESHOLD) {
            MaskIter::Indices(indices) => filter_indices(array, indices.iter().copied()),
            MaskIter::Slices(slices) => filter_slices(array, slices.iter().copied()),
        }?;

        // SAFETY: Filter operation preserves the dtype of each chunk.
        // All filtered chunks maintain the same dtype as the original array.
        let result =
            unsafe { ChunkedArray::new_unchecked(chunks, array.dtype().clone()) }.into_array();

        Ok(Some(result))
    }
}
