// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Canonical;
use crate::arrays::extension::ExtensionArray;
use crate::arrays::extension::ExtensionVTable;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<ExtensionVTable> for ExtensionVTable {
    fn canonicalize(array: &ExtensionArray) -> VortexResult<Canonical> {
        Ok(Canonical::Extension(array.clone()))
    }
}
