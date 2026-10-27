// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::arrays::Primitive;
use crate::arrays::VarBin;
use crate::arrays::varbin::VarBinArraySlotsExt;
use crate::arrays::varbin::VarBinSlots;
use crate::arrays::varbin::varbin_scalar;
use crate::match_each_integer_ptype;
use crate::scalar::Scalar;

impl OperationsVTable<VarBin> for VarBin {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, VarBin>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        if !state.is_valid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }

        let start = offset_at(state, index, ctx)?;
        let end = offset_at(state, index + 1, ctx)?;
        Ok(varbin_scalar(
            array.bytes().slice(start..end),
            array.dtype(),
        ))
    }

    fn scalar_at(
        array: ArrayView<'_, VarBin>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

fn offset_at(
    state: &mut ProbeState<'_, VarBin>,
    index: usize,
    ctx: &mut ExecutionCtx,
) -> VortexResult<usize> {
    if let Some(primitive) = state.array().offsets().as_opt::<Primitive>()
        && primitive.buffer_handle().is_on_host()
    {
        return match_each_integer_ptype!(primitive.ptype(), |T| {
            Ok(usize::try_from(primitive.as_slice::<T>()[index])
                .vortex_expect("Failed to convert offset to usize"))
        });
    }

    let offset = state
        .slot(VarBinSlots::OFFSETS)?
        .ok_or_else(|| vortex_err!("VarBin offsets slot is missing"))?
        .execute_scalar(index, ctx)?;
    Ok((&offset)
        .try_into()
        .vortex_expect("Failed to convert offset to usize"))
}
