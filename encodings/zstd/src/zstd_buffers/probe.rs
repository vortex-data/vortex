// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::ProbeState;
use vortex_array::RepeatedArrayProbe;
use vortex_array::scalar::Scalar;
use vortex_array::vtable::OperationsVTable;
use vortex_error::VortexResult;

use super::ZstdBuffers;

/// Every read decompresses all of the buffers and rebuilds the wrapped array, so a repeated probe
/// keeps one probe over the rebuilt array.
#[derive(Default)]
pub struct ZstdBuffersProbeState {
    inner: Option<RepeatedArrayProbe>,
}

impl OperationsVTable<ZstdBuffers> for ZstdBuffers {
    type ProbeState = ZstdBuffersProbeState;

    fn probe_scalar(
        state: &mut ProbeState<'_, ZstdBuffers>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        if !state.is_valid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }

        // A one-off read is really slow. Execute the array into canonical form first, or read it
        // through a repeated probe.
        let Some(retained) = state.retained() else {
            let inner =
                ZstdBuffers::decompress_and_build_inner(&array.into_owned(), ctx.session())?;
            return inner.execute_scalar(index, ctx);
        };

        let inner = match &mut retained.inner {
            Some(inner) => inner,
            slot @ None => slot.insert(RepeatedArrayProbe::new(
                ZstdBuffers::decompress_and_build_inner(&array.into_owned(), ctx.session())?,
            )),
        };
        inner.execute_scalar(index, ctx)
    }

    fn scalar_at(
        array: ArrayView<'_, ZstdBuffers>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}
