// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::arrays::Extension;
use crate::arrays::extension::ExtensionArrayExt;
use crate::arrays::extension::ExtensionSlots;
use crate::scalar::Scalar;

impl OperationsVTable<Extension> for Extension {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, Extension>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let ext_dtype = state.array().ext_dtype().clone();
        let storage = state
            .slot(ExtensionSlots::STORAGE)?
            .ok_or_else(|| vortex_err!("Extension storage slot is missing"))?
            .execute_scalar(index, ctx)?;
        Ok(Scalar::extension_ref(ext_dtype, storage))
    }

    fn scalar_at(
        array: ArrayView<'_, Extension>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}
