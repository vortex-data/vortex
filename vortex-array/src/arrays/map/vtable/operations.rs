// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::arrays::Map;
use crate::arrays::map::MapSlots;
use crate::scalar::Scalar;

impl OperationsVTable<Map> for Map {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, Map>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let dtype = state.array().dtype().clone();
        let (_, entries) = state
            .slot(MapSlots::ENTRIES)?
            .ok_or_else(|| vortex_err!("Map entries slot is missing"))?
            .execute_scalar(index, ctx)?
            .into_parts();
        // The validated ListView child and Map scalar share the same entry tuple layout.
        Scalar::try_new(dtype, entries)
    }

    fn scalar_at(
        array: ArrayView<'_, Map>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}
