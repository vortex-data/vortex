// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_array::ArrayRef;
use vortex_array::ArrayVTable;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::Dict;
use vortex_array::arrays::Filter;
use vortex_array::arrays::Slice;
use vortex_array::arrays::dict::TakeExecuteAdaptor;
use vortex_array::arrays::filter::FilterExecuteAdaptor;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::optimizer::kernels::ArrayKernelsExt;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::binary::CompareExecuteAdaptor;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::RunEnd;
use crate::array::RunEndArrayExt;
use crate::compute::take_from::RunEndTakeFrom;

pub(super) fn initialize(session: &VortexSession) {
    let Some(kernels) = session.kernels() else {
        return;
    };
    kernels.register_execute_parent_kernel(Binary.id(), RunEnd, CompareExecuteAdaptor(RunEnd));
    kernels.register_execute_parent_kernel(Slice.id(), RunEnd, RunEndSliceKernel);
    kernels.register_execute_parent_kernel(Filter.id(), RunEnd, FilterExecuteAdaptor(RunEnd));
    kernels.register_execute_parent_kernel(Dict.id(), RunEnd, TakeExecuteAdaptor(RunEnd));
    kernels.register_execute_parent_kernel(Dict.id(), RunEnd, RunEndTakeFrom);
}

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
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        slice(array, parent.slice_range().clone(), ctx).map(Some)
    }
}

fn slice(
    array: ArrayView<'_, RunEnd>,
    range: Range<usize>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let new_length = range.len();

    let slice_begin = array.find_physical_index(range.start)?;
    let slice_end = crate::ops::find_slice_end_index(array.ends(), range.end + array.offset())?;

    // If the sliced range contains only a single run, opt to return a ConstantArray.
    if slice_begin + 1 == slice_end {
        let value = array.values().execute_scalar(slice_begin, ctx)?;
        return Ok(ConstantArray::new(value, new_length).into_array());
    }

    // SAFETY: we maintain the ends invariant in our slice implementation
    Ok(unsafe {
        RunEnd::new_unchecked(
            array.ends().slice(slice_begin..slice_end)?,
            array.values().slice(slice_begin..slice_end)?,
            range.start + array.offset(),
            new_length,
        )
        .into_array()
    })
}
