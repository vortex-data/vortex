// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::ProbeState;
use vortex_array::RepeatedArrayProbe;
use vortex_array::arrays::varbin::varbin_scalar;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::FSST;
use crate::FSSTArrayExt;

/// The codes array is rebuilt from the codes buffer and the offsets slot on every read, so a
/// repeated probe keeps one probe over it and the offsets child keeps its preparation.
#[derive(Default)]
pub struct FsstProbeState {
    codes: Option<RepeatedArrayProbe>,
}

impl OperationsVTable<FSST> for FSST {
    type ProbeState = FsstProbeState;

    fn probe_scalar(
        state: &mut ProbeState<'_, FSST>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        let compressed = match state.retained() {
            None => array.codes().into_array().execute_scalar(index, ctx)?,
            Some(retained) => retained
                .codes
                .get_or_insert_with(|| RepeatedArrayProbe::new(array.codes().into_array()))
                .execute_scalar(index, ctx)?,
        };
        let binary_datum = compressed.as_binary().value().vortex_expect("non-null");

        let decoded_buffer = ByteBuffer::from(array.decompressor().decompress(binary_datum));
        Ok(varbin_scalar(decoded_buffer, array.dtype()))
    }

    fn scalar_at(
        array: ArrayView<'_, FSST>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}
