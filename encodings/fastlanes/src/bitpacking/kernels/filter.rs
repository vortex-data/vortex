// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ExecutionCtx;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::FilterVTable;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::matchers::Exact;
use vortex_error::VortexResult;
use vortex_vector::Vector;

use crate::BitPackedArray;
use crate::BitPackedVTable;

#[derive(Debug)]
struct BitPackingFilterKernel;

impl ExecuteParentKernel<BitPackedVTable> for BitPackingFilterKernel {
    type Parent = Exact<FilterVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::from(&FilterVTable)
    }

    fn execute_parent(
        &self,
        array: &BitPackedArray,
        parent: &FilterArray,
        child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Vector>> {
        todo!()
    }
}
