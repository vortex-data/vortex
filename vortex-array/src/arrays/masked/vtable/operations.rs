// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::arrays::Masked;
use crate::arrays::masked::MaskedSlots;
use crate::scalar::Scalar;

impl OperationsVTable<Masked> for Masked {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, Masked>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // Invalid indices are handled by the entrypoint function.
        Ok(state
            .slot(MaskedSlots::CHILD)?
            .ok_or_else(|| vortex_err!("Masked child slot is missing"))?
            .execute_scalar(index, ctx)?
            .into_nullable())
    }

    fn scalar_at(
        array: ArrayView<'_, Masked>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}
