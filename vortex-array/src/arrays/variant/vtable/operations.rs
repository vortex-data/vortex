// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use super::merge_typed_scalar_as_variant;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::arrays::Variant;
use crate::arrays::variant::VariantSlots;
use crate::scalar::Scalar;

impl OperationsVTable<Variant> for Variant {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, Variant>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        if state
            .slot(VariantSlots::CORE_STORAGE)?
            .vortex_expect("validated Variant core storage slot")
            .execute_is_invalid(index, ctx)?
        {
            return Ok(Scalar::null(state.array().dtype().clone()));
        }

        let Some(mut shredded) = state.slot(VariantSlots::SHREDDED)? else {
            return state
                .slot(VariantSlots::CORE_STORAGE)?
                .vortex_expect("validated Variant core storage slot")
                .execute_scalar(index, ctx);
        };

        let typed = shredded.execute_scalar(index, ctx)?;
        // If the shredded value is null OR we shredded an object we want to merge back together.
        let fallback = (typed.is_null() || typed.dtype().is_struct())
            .then(|| {
                state
                    .slot(VariantSlots::CORE_STORAGE)?
                    .vortex_expect("validated Variant core storage slot")
                    .execute_scalar(index, ctx)
            })
            .transpose()?;
        merge_typed_scalar_as_variant(typed, fallback, state.array().dtype())
    }

    fn scalar_at(
        array: ArrayView<'_, Variant>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}
