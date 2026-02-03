// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::SliceVTable;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::matchers::Exact;
use vortex_error::VortexResult;

use crate::RunEndArray;
use crate::RunEndVTable;

pub(super) const PARENT_KERNELS: ParentKernelSet<RunEndVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&RunEndSliceKernel)]);

/// Kernel to execute slicing on a RunEnd array.
///
/// This directly slices the RunEnd array using OperationsVTable::slice and then
/// executes the result to get the canonical form.
#[derive(Debug)]
struct RunEndSliceKernel;

impl ExecuteParentKernel<RunEndVTable> for RunEndSliceKernel {
    type Parent = Exact<SliceVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::new()
    }

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

    let slice_begin = array.find_physical_index(range.start)?;
    let slice_end = crate::ops::find_slice_end_index(array.ends(), range.end + array.offset())?;

    // If the sliced range contains only a single run, opt to return a ConstantArray.
    if slice_begin + 1 == slice_end {
        let value = array.values().scalar_at(slice_begin)?;
        return Ok(ConstantArray::new(value, new_length).into_array());
    }

    // SAFETY: we maintain the ends invariant in our slice implementation
    Ok(unsafe {
        RunEndArray::new_unchecked(
            array.ends().slice(slice_begin..slice_end)?,
            array.values().slice(slice_begin..slice_end)?,
            range.start + array.offset(),
            new_length,
        )
        .into_array()
    })
}
