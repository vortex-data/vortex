// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Array;
use crate::arrays::extension::ExtensionArray;
use crate::arrays::extension::ExtensionVTable;
use crate::vtable::ValidityChild;

impl ValidityChild<ExtensionVTable> for ExtensionVTable {
    fn validity_child(array: &ExtensionArray) -> &dyn Array {
        array.storage.as_ref()
    }
}
