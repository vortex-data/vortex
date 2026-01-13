// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::FilterArray;
use vortex_array::arrays::FilterVTable;
use vortex_array::kernel::ExecuteParentKernel;
use vortex_array::matchers::Exact;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::SparseArray;
use crate::SparseVTable;

#[derive(Debug)]
pub(super) struct SparseFilterKernel;

impl ExecuteParentKernel<SparseVTable> for SparseFilterKernel {
    type Parent = Exact<FilterVTable>;

    fn parent(&self) -> Self::Parent {
        Exact::from(&FilterVTable)
    }

    fn execute_parent(
        &self,
        array: &SparseArray,
        parent: &FilterArray,
        _child_idx: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Canonical>> {
        let mask = parent.filter_mask();

        // Early return for trivial masks
        match mask {
            Mask::AllTrue(_) | Mask::AllFalse(_) => return Ok(None),
            Mask::Values(_) => {}
        }

        let new_length = mask.true_count();

        let Some(new_patches) = array.patches().filter(mask)? else {
            return Ok(Some(
                ConstantArray::new(array.fill_scalar().clone(), new_length).to_canonical(),
            ));
        };

        Ok(Some(
            SparseArray::try_new_from_patches(new_patches, array.fill_scalar().clone())?
                .to_canonical(),
        ))
    }
}
