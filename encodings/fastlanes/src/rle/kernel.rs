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

use crate::RLEArray;
use crate::RLEVTable;

pub(super) const PARENT_KERNELS: ParentKernelSet<RLEVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&RLESliceKernel)]);

/// Kernel to execute slicing on an RLE array.
///
/// This directly slices the RLE array using OperationsVTable::slice and then
/// executes the result to get the canonical form.
#[derive(Debug)]
struct RLESliceKernel;

impl ExecuteParentKernel<RLEVTable> for RLESliceKernel {
    type Parent = Exact<SliceVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::from(&SliceVTable)
    }

    fn execute_parent(
        &self,
        array: &RLEArray,
        parent: &SliceArray,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Canonical>> {
        let sliced = RLEVTable::slice(array, parent.slice_range().clone());

        sliced.execute::<Canonical>(ctx).map(Some)
    }
}
