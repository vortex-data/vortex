// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Slice;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::RunEndArray;
use crate::compute::take_from::RunEndTakeFrom;

pub(super) const PARENT_KERNELS: ParentKernelSet<RunEnd> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&CompareExecuteAdaptor(RunEnd)),
    ParentKernelSet::lift(&RunEndSliceKernel),
    ParentKernelSet::lift(&FilterExecuteAdaptor(RunEnd)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(RunEnd)),
    ParentKernelSet::lift(&RunEndTakeFrom),
]);

/// Kernel to execute slicing on a RunEnd array.
///
/// This directly slices the RunEnd array using OperationsVTable::slice and then
/// executes the result to get the canonical form.
#[derive(Debug)]
struct RunEndSliceKernel;

impl ExecuteParentKernel<RunEnd> for RunEndSliceKernel {
    type Parent = Slice;

    fn execute_parent(
        &self,
        array: &RunEndArray,
        parent: &SliceArray,
        _child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        slice(array, parent.slice_range().clone()).map(Some)
    }
}

fn slice(array: &RunEndArray, range: Range<usize>) -> VortexResult<ArrayRef> {
    let new_length = range.len();

    let (raw_ends, offset) = array.raw_ends_and_offset();
    let slice_begin = array.find_physical_index(range.start)?;
    let slice_end = crate::ops::find_slice_end_index(raw_ends, range.end + offset)?;

    // If the sliced range contains only a single run, opt to return a ConstantArray.
    if slice_begin + 1 == slice_end {
        let value = array.values().scalar_at(slice_begin)?;
        return Ok(ConstantArray::new(value, new_length).into_array());
    }

    let new_offset = range.start + offset;
    let sliced_raw_ends = raw_ends.slice(slice_begin..slice_end)?;
    let new_ends = vortex_array::patches::wrap_with_offset(sliced_raw_ends, new_offset)?;

    // SAFETY: we maintain the ends invariant in our slice implementation
    Ok(unsafe {
        RunEndArray::new_unchecked(
            new_ends,
            array.values().slice(slice_begin..slice_end)?,
            new_length,
        )
        .into_array()
    })
}
