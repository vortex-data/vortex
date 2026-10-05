// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use super::Dict;
use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::arrays::dict::DictSlots;
use crate::scalar::Scalar;

impl OperationsVTable<Dict> for Dict {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, Dict>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let dtype = state.array().dtype().clone();
        let code = state
            .slot(DictSlots::CODES)?
            .ok_or_else(|| vortex_err!("Dict codes slot is missing"))?
            .execute_scalar(index, ctx)?;
        let Some(dict_index) = code.as_primitive().as_::<usize>() else {
            return Ok(Scalar::null(dtype));
        };

        Ok(state
            .slot(DictSlots::VALUES)?
            .ok_or_else(|| vortex_err!("Dict values slot is missing"))?
            .execute_scalar(dict_index, ctx)?
            .cast(&dtype)
            .vortex_expect("Array dtype will only differ by nullability"))
    }

    fn scalar_at(
        array: ArrayView<'_, Dict>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}
