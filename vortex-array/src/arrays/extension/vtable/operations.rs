// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::DynArray;
use crate::arrays::extension::ExtensionArray;
use crate::arrays::extension::ExtensionVTable;
use crate::scalar::Scalar;
use crate::vtable::OperationsVTable;

impl OperationsVTable<ExtensionVTable> for ExtensionVTable {
    fn scalar_at(array: &ExtensionArray, index: usize) -> VortexResult<Scalar> {
        Ok(Scalar::extension_ref(
            array.ext_dtype().clone(),
            array.storage().scalar_at(index)?,
        ))
    }
}
