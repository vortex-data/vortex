// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::arrays::Extension;
use crate::scalar::Scalar;
use crate::vtable::Array;
use crate::vtable::OperationsVTable;

impl OperationsVTable<Extension> for Extension {
    fn scalar_at(
        array: &Array<Extension>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::extension_ref(
            array.ext_dtype().clone(),
            array.storage_array().scalar_at(index)?,
        ))
    }
}
