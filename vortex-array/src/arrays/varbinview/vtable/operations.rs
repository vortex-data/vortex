// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::arrays::VarBinView;
use crate::arrays::varbin::varbin_scalar_unchecked;
use crate::scalar::Scalar;

impl OperationsVTable<VarBinView> for VarBinView {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, VarBinView>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        if !state.is_valid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }
        // SAFETY: `VarBinViewArray` has a Utf8 or Binary dtype, and it checks that all Utf8 data
        // is valid UTF-8.
        Ok(unsafe { varbin_scalar_unchecked(array.bytes_at(index), array.dtype()) })
    }

    fn scalar_at(
        array: ArrayView<'_, VarBinView>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}
