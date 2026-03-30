// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::DynArray;
use crate::ExecutionCtx;
use crate::arrays::Masked;
use crate::arrays::MaskedArray;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<Masked> for Masked {
    fn scalar_at(
        array: &MaskedArray,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        // Invalid indices are handled by the entrypoint function.
        Ok(array.child().scalar_at(index)?.into_nullable())
    }
}
