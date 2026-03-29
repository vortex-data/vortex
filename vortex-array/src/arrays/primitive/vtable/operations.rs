// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::arrays::Primitive;
use crate::match_each_native_ptype;
use crate::scalar::Scalar;
use crate::vtable::Array;
use crate::vtable::OperationsVTable;

impl OperationsVTable<Primitive> for Primitive {
    fn scalar_at(
        array: &Array<Primitive>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(match_each_native_ptype!(array.ptype(), |T| {
            Scalar::primitive(array.as_slice::<T>()[index], array.dtype().nullability())
        }))
    }
}
