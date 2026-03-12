// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayRef;
use crate::arrays::ExtensionArray;
use crate::arrays::ExtensionVTable;
use crate::vtable::ValidityChild;

impl ValidityChild<ExtensionVTable> for ExtensionVTable {
    fn validity_child(array: &ExtensionArray) -> &ArrayRef {
        array.storage_array()
    }
}
