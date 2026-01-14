// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::SliceArray;
use vortex_array::arrays::SliceVTable;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::matchers::Exact;
use vortex_array::vtable::OperationsVTable;
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
        Exact::from(&SliceVTable)
    }

    fn execute_parent(
        &self,
        array: &RunEndArray,
        parent: &SliceArray,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Canonical>> {
        let sliced = RunEndVTable::slice(array, parent.slice_range().clone());

        sliced.execute::<Canonical>(ctx).map(Some)
    }
}
