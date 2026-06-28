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
use vortex_array::dtype::Nullability;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::not::NotExecuteAdaptor;
use vortex_error::VortexResult;
use vortex_runend::find_slice_end_index;

use crate::RunEndBool;
use crate::array::RunEndBoolArrayExt;
use crate::compress::value_at_index;

pub(super) const PARENT_KERNELS: ParentKernelSet<RunEndBool> = ParentKernelSet::new(&[
    ParentKernelSet::lift(&RunEndBoolSliceKernel),
    ParentKernelSet::lift(&FilterExecuteAdaptor(RunEndBool)),
    ParentKernelSet::lift(&TakeExecuteAdaptor(RunEndBool)),
    ParentKernelSet::lift(&NotExecuteAdaptor(RunEndBool)),
]);

/// Kernel to execute slicing on a [`RunEndBool`] array.
#[derive(Debug)]
struct RunEndBoolSliceKernel;

impl ExecuteParentKernel<RunEndBool> for RunEndBoolSliceKernel {
    type Parent = Slice;

    fn execute_parent(
        &self,
        array: ArrayView<'_, RunEndBool>,
        parent: ArrayView<'_, Slice>,
        _child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        slice(array, parent.slice_range().clone()).map(Some)
    }
}

fn slice(array: ArrayView<'_, RunEndBool>, range: Range<usize>) -> VortexResult<ArrayRef> {
    let new_length = range.len();

    let slice_begin = array.find_physical_index(range.start)?;
    let slice_end = find_slice_end_index(array.ends(), range.end + array.offset())?;

    let nullability = array.nullability();
    let sliced_validity = array.bool_validity().slice(range.start..range.end)?;

    // If the sliced range contains only a single run and the array is non-nullable, opt to return a
    // ConstantArray. When nullable we keep the run-end structure so per-element validity is
    // preserved by canonicalization.
    if slice_begin + 1 == slice_end && nullability == Nullability::NonNullable {
        let value = value_at_index(slice_begin, array.start());
        return Ok(ConstantArray::new(Scalar::bool(value, nullability), new_length).into_array());
    }

    let new_start = value_at_index(slice_begin, array.start());

    // SAFETY: slicing preserves the strictly-increasing ends invariant.
    Ok(unsafe {
        RunEndBool::new_unchecked(
            array.ends().slice(slice_begin..slice_end)?,
            new_start,
            range.start + array.offset(),
            new_length,
            sliced_validity,
        )
        .into_array()
    })
}
