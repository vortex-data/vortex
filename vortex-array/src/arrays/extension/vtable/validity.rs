// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayRef;
use crate::arrays::Extension;
use crate::arrays::ExtensionArray;
use crate::vtable::ValidityChild;

impl ValidityChild<Extension> for Extension {
    fn validity_child(array: &ExtensionArray) -> &ArrayRef {
        array.storage_array()
    }
}
