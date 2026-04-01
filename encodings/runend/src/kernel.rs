// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Slice;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_error::VortexResult;

use crate::RunEnd;
use crate::RunEndData;
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
        array: ArrayView<'_, RunEnd>,
        parent: ArrayView<'_, Slice>,
        _child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        slice(&array, parent.slice_range().clone()).map(Some)
    }
}

fn slice(array: &RunEndData, range: Range<usize>) -> VortexResult<ArrayRef> {
    let new_length = range.len();

    let slice_begin = array.find_physical_index(range.start)?;
    let slice_end = crate::ops::find_slice_end_index(array.ends(), range.end + array.offset())?;

    // If the sliced range contains only a single run, opt to return a ConstantArray.
    if slice_begin + 1 == slice_end {
        let value = array.values().scalar_at(slice_begin)?;
        return Ok(ConstantArray::new(value, new_length).into_array());
    }

    // SAFETY: we maintain the ends invariant in our slice implementation
    Ok(unsafe {
        RunEndData::new_unchecked(
            array.ends().slice(slice_begin..slice_end)?,
            array.values().slice(slice_begin..slice_end)?,
            range.start + array.offset(),
            new_length,
        )
        .into_array()
    })
}
