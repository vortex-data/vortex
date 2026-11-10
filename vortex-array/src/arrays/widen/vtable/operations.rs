// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::array::ProbeState;
use crate::arrays::Widen;
use crate::arrays::widen::WidenSlots;
use crate::dtype::DType;
use crate::scalar::Scalar;

impl OperationsVTable<Widen> for Widen {
    type ProbeState = ();

    fn probe_scalar(
        state: &mut ProbeState<'_, Widen>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        let array = state.array();
        let value = state
            .slot(WidenSlots::VALUES)?
            .ok_or_else(|| vortex_err!("Widen values slot is missing"))?
            .execute_scalar(index, ctx)?;

        decode_narrow_scalar(&value, array.dtype())
    }

    fn scalar_at(
        array: ArrayView<'_, Widen>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Self::probe_scalar(&mut ProbeState::once(array), index, ctx)
    }
}

/// Decodes `value`, a scalar of the values of a `Widen`, to its `logical` dtype.
///
/// The values have the nullability of the logical dtype, so a null value decodes to a null.
fn decode_narrow_scalar(value: &Scalar, logical: &DType) -> VortexResult<Scalar> {
    value.cast(logical)
}
