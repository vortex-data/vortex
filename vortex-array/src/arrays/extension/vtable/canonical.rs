// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Canonical;
use crate::arrays::extension::{
    ExtensionArray,
    ExtensionVTable,
};
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<ExtensionVTable> for ExtensionVTable {
    fn canonicalize(array: &ExtensionArray) -> Canonical {
        Canonical::Extension(array.clone())
    }
}
