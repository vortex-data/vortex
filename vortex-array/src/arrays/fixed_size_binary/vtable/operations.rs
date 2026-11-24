// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::arrays::FixedSizeBinary;
use crate::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

impl OperationsVTable<FixedSizeBinary> for FixedSizeBinary {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, FixedSizeBinary>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        if !state.is_valid(index, ctx)? {
            return Ok(Scalar::null(array.dtype().clone()));
        }

        let value = ByteBuffer::copy_from(array.value(index));
        // SAFETY: Every row of a valid array holds exactly `byte_width` bytes.
        Ok(unsafe {
            Scalar::new_unchecked(array.dtype().clone(), Some(ScalarValue::Binary(value)))
        })
    }

    fn scalar_at(
        array: ArrayView<'_, FixedSizeBinary>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}
