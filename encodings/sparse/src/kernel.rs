// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayRef;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::FilterVTable;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::kernel::ParentKernelSet;
use vortex_array::matchers::Exact;
use vortex_error::VortexResult;

use crate::SparseArray;
use crate::SparseVTable;

pub(super) const PARENT_KERNELS: ParentKernelSet<SparseVTable> =
    ParentKernelSet::new(&[ParentKernelSet::lift(&SparseFilterKernel)]);

#[derive(Debug)]
pub(super) struct SparseFilterKernel;

impl ExecuteParentKernel<SparseVTable> for SparseFilterKernel {
    type Parent = Exact<FilterVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::new()
    }

    fn execute_parent(
        &self,
        array: &SparseArray,
        parent: &FilterArray,
        _child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let mask = parent.filter_mask();
        let new_length = mask.true_count();

        let Some(new_patches) = array.patches().filter(mask)? else {
            return Ok(Some(
                ConstantArray::new(array.fill_scalar().clone(), new_length).into_array(),
            ));
        };

        Ok(Some(
            SparseArray::try_new_from_patches(new_patches, array.fill_scalar().clone())?
                .into_array(),
        ))
    }
}
