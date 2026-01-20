// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_scalar::Scalar;

use crate::arrays::extension::ExtensionArray;
use crate::arrays::extension::ExtensionVTable;
use crate::vtable::OperationsVTable;

impl OperationsVTable<ExtensionVTable> for ExtensionVTable {
    fn scalar_at(array: &ExtensionArray, index: usize) -> Scalar {
        Scalar::extension(array.ext_dtype().clone(), array.storage().scalar_at(index))
    }
}
