// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::slice::SliceExecuteAdaptor;
use vortex_array::arrays::slice::SliceKernel;
use vortex_array::kernel::ParentKernelSet;
use vortex_error::VortexResult;

use crate::FL_CHUNK_SIZE;
use crate::RLE;
use crate::RLEData;

pub(crate) static PARENT_KERNELS: ParentKernelSet<RLE> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&SliceExecuteAdaptor(RLE))]);

impl SliceKernel for RLE {
    fn slice(
        array: ArrayView<'_, Self>,
        range: Range<usize>,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let offset_in_chunk = array.offset();
        let chunk_start_idx = (offset_in_chunk + range.start) / FL_CHUNK_SIZE;
        let chunk_end_idx = (offset_in_chunk + range.end).div_ceil(FL_CHUNK_SIZE);

        let values_start_idx = array.values_idx_offset(chunk_start_idx);
        let values_end_idx = if chunk_end_idx < array.values_idx_offsets().len() {
            array.values_idx_offset(chunk_end_idx)
        } else {
            array.values().len()
        };

        let sliced_values = array.values().slice(values_start_idx..values_end_idx)?;

        let sliced_values_idx_offsets = array
            .values_idx_offsets()
            .slice(chunk_start_idx..chunk_end_idx)?;

        let sliced_indices = array
            .indices()
            .slice(chunk_start_idx * FL_CHUNK_SIZE..chunk_end_idx * FL_CHUNK_SIZE)?;

        // SAFETY: Slicing preserves all invariants.
        Ok(Some(unsafe {
            RLEData::new_unchecked(
                sliced_values,
                sliced_indices,
                sliced_values_idx_offsets,
                array.dtype().clone(),
                // Keep the offset relative to the first chunk.
                (array.offset() + range.start) % FL_CHUNK_SIZE,
                range.len(),
            )
            .into_array()
        }))
    }
}
