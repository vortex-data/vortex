// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::FloatPatchStrategy;
use super::IntegerPatchStrategy;
use super::PatchStrategy;
use super::Patched;
use super::PatchedArrayExt;
use super::PatchedArraySlotsExt;
use crate::ArrayRef;
use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::arrays::BoolArray;
use crate::arrays::ConstantArray;
use crate::arrays::dict::TakeExecute;
use crate::arrays::filter::FilterKernel;
use crate::arrays::scalar_fn::ScalarFnValidity;
use crate::builtins::ArrayBuiltins;
use crate::kernel::ExecuteParentKernel;
use crate::patches::Patches;
use crate::scalar_fn::builtin::compare::CompareKernel;
use crate::scalar_fn::builtin::operators::CompareOperator;

pub(super) fn select_rows(
    array: ArrayView<'_, Patched>,
    base: ArrayRef,
    patches: Option<Patches>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    Ok(Some(match patches {
        Some(patches) => {
            Patched::with_strategy(base, &patches, array.strategy(), ctx)?.into_array()
        }
        None => base,
    }))
}

impl FilterKernel for Patched {
    fn filter(
        array: ArrayView<'_, Self>,
        mask: &Mask,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let base = array.inner().filter(mask.clone())?;
        select_rows(array, base, array.patches().filter(mask, ctx)?, ctx)
    }
}

impl TakeExecute for Patched {
    fn take(
        array: ArrayView<'_, Self>,
        indices: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let base = array.inner().take(indices.clone())?;
        // The base take already supplies nulls for null indices.
        select_rows(array, base, array.patches().take(indices, ctx)?, ctx)
    }
}

impl CompareKernel for Patched {
    fn compare(
        array: ArrayView<'_, Self>,
        rhs: &ArrayRef,
        op: CompareOperator,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<ArrayRef>> {
        let (PatchStrategy::Integer(IntegerPatchStrategy::Replace)
        | PatchStrategy::Float(FloatPatchStrategy::Replace)) = array.strategy();
        let Some(constant) = rhs.as_constant() else {
            return Ok(None);
        };
        let base = array
            .inner()
            .compare(rhs.clone(), op)?
            .execute::<BoolArray>(ctx)?;
        let patches = array.patches().map_values(|values| {
            let rhs = ConstantArray::new(constant.clone(), values.len()).into_array();
            values.compare(rhs, op)
        })?;
        Ok(Some(base.patch(&patches, ctx)?.into_array()))
    }
}

/// Project replacement validity directly from the base and patch validity masks.
#[derive(Debug)]
pub(super) struct PatchValidity;

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
