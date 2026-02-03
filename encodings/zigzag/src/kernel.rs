// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::FilterVTable;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::matchers::Exact;
use vortex_error::VortexResult;

use crate::ZigZagArray;
use crate::ZigZagVTable;

pub(super) const PARENT_KERNELS: ParentKernelSet<ZigZagVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&ZigZagFilterKernel)]);

#[derive(Debug)]
struct ZigZagFilterKernel;

impl ExecuteParentKernel<ZigZagVTable> for ZigZagFilterKernel {
    type Parent = Exact<FilterVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::new()
    }

    fn execute_parent(
        &self,
        array: &ZigZagArray,
        parent: &FilterArray,
        _child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let mask = parent.filter_mask();
        let encoded = array.encoded().filter(mask.clone())?;
        Ok(Some(ZigZagArray::try_new(encoded)?.into_array()))
    }
}
