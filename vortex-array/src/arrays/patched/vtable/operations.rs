// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::ArrayView;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::array::ValidityVTable;
use crate::arrays::Primitive;
use crate::arrays::patched::FloatPatchStrategy;
use crate::arrays::patched::IntegerPatchStrategy;
use crate::arrays::patched::PATCH_SLOTS;
use crate::arrays::patched::PatchStrategy;
use crate::arrays::patched::Patched;
use crate::arrays::patched::PatchedArrayExt;
use crate::arrays::patched::PatchedArraySlotsExt;
use crate::arrays::patched::PatchedSlots;
use crate::arrays::scalar_fn::ValidityArray;
use crate::match_each_unsigned_integer_ptype;
use crate::patches::PatchesProbeState;
use crate::scalar::Scalar;
use crate::validity::Validity;

impl OperationsVTable<Patched> for Patched {
    type ProbeState = PatchesProbeState;

    fn probe_scalar(
        state: &mut ProbeState<'_, Self>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        let (PatchStrategy::Integer(IntegerPatchStrategy::Replace)
        | PatchStrategy::Float(FloatPatchStrategy::Replace)) = array.strategy();
        if let Some(indices) = array.patch_indices().as_opt::<Primitive>()
            && indices.buffer_handle().is_on_host()
        {
            // Search the existing buffer directly, without decoding or slicing a chunk per probe.
            let row = array.data().patches.offset() + index;
            let position = match_each_unsigned_integer_ptype!(indices.ptype(), |I| {
                I::try_from(row)
                    .ok()
                    .and_then(|needle| indices.as_slice::<I>().binary_search(&needle).ok())
            });
            if let Some(position) = position {
                return state
                    .slot(PatchedSlots::PATCH_VALUES)?
                    .vortex_expect("required patch values")
                    .execute_scalar(position, ctx);
            }
        } else if let Some(value) = array
            .patches()
            .probe_patched(state, PATCH_SLOTS, index, ctx)?
        {
            return Ok(value);
        }
        state
            .slot(PatchedSlots::INNER)?
            .vortex_expect("required base")
            .execute_scalar(index, ctx)
    }

    fn scalar_at(
        array: ArrayView<'_, Self>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

impl ValidityVTable<Patched> for Patched {
    fn validity(array: ArrayView<'_, Self>) -> VortexResult<Validity> {
        let (PatchStrategy::Integer(IntegerPatchStrategy::Replace)
        | PatchStrategy::Float(FloatPatchStrategy::Replace)) = array.strategy();
        match (array.inner().validity()?, array.patch_values().validity()?) {
            (Validity::AllValid, Validity::AllValid) => Ok(Validity::AllValid),
            (Validity::AllInvalid, Validity::AllInvalid) => Ok(Validity::AllInvalid),
            _ => Ok(Validity::Array(
                ValidityArray::new(array.array().clone()).into_array(),
            )),
        }
    }
}
