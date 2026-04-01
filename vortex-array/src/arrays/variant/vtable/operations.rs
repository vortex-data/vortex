// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::arrays::Variant;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<Variant> for Variant {
    fn scalar_at(
        array: &<Variant as crate::vtable::VTable>::Array,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        array.child().scalar_at(index)
    }
}
