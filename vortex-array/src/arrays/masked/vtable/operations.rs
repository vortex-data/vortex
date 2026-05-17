// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Masked;
use crate::arrays::masked::MaskedArraySlotsExt;
use crate::scalar::Scalar;

impl OperationsVTable<Masked> for Masked {
    // Transparent wrapper that defers to a child; skip wrapper-level cache.
    const FAST_SCALAR_AT: bool = true;

    fn scalar_at(
        array: ArrayView<'_, Masked>,
        index: usize,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // Invalid indices are handled by the entrypoint function.
        Ok(array.child().execute_scalar(index, ctx)?.into_nullable())
    }
}
