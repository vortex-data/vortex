// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::ExecutionCtx;
use crate::array::ArrayView;
use crate::array::OperationsVTable;
use crate::arrays::Extension;
use crate::scalar::Scalar;

impl OperationsVTable<Extension> for Extension {
    fn scalar_at(
        array: ArrayView<'_, Extension>,
        index: usize,
        _ctx: &mut ExecutionCtx,
    ) -> VortexResult<Scalar> {
        Ok(Scalar::extension_ref(
            array.ext_dtype().clone(),
            array.storage_array().scalar_at(index)?,
        ))
    }
}
