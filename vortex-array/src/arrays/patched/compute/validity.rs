// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::arrays::patched::FloatPatchStrategy;
use crate::arrays::patched::IntegerPatchStrategy;
use crate::arrays::patched::PatchStrategy;
use crate::arrays::patched::Patched;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::arrays::scalar_fn::ScalarFnValidity;
use crate::kernel::ExecuteParentKernel;

/// Project replacement validity directly from the base and patch validity masks.
#[derive(Debug)]
pub(in crate::arrays::patched) struct PatchValidity;

impl ExecuteParentKernel<Patched> for PatchValidity {
    type Parent = ScalarFnValidity;

    fn execute_parent(
        &self,
        array: ArrayView<'_, Patched>,
        _parent: ArrayView<'_, ScalarFnValidity>,
        _child_idx: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let (PatchStrategy::Integer(IntegerPatchStrategy::Replace)
        | PatchStrategy::Float(FloatPatchStrategy::Replace)) = array.strategy();
        let validity = array.inner().validity()?.patch(
            array.len(),
            array.data().patches.offset(),
            array.patch_indices(),
            &array.patch_values().validity()?,
            ctx,
        )?;
        Ok(Some(validity.to_array(array.len())))
    }
}
