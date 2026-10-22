// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::ArrayVTable;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::ConstantArray;
use crate::arrays::Dict;
use crate::arrays::Filter;
use crate::arrays::Slice;
use crate::arrays::dict::TakeExecuteAdaptor;
use crate::arrays::filter::FilterExecuteAdaptor;
use crate::arrays::runend::RunEnd;
use crate::arrays::runend::array::RunEndArrayExt;
use crate::arrays::runend::array::RunEndArraySlotsExt;
use crate::arrays::runend::compute::take_from::RunEndTakeFrom;
use crate::kernel::ExecuteParentKernel;
use crate::optimizer::kernels::ArrayKernelsExt;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::binary::CompareExecuteAdaptor;

pub(super) fn initialize(session: &VortexSession) {
    let kernels = session.kernels();
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

    let slice_begin = array.find_physical_index(range.start, ctx)?;
    let slice_end = array.find_slice_end_index(range.end, ctx)?;

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
