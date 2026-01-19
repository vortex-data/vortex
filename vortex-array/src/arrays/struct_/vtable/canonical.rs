// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Canonical;
use crate::arrays::struct_::StructArray;
use crate::arrays::struct_::StructVTable;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<StructVTable> for StructVTable {
    fn canonicalize(array: &StructArray) -> VortexResult<Canonical> {
        Ok(Canonical::Struct(array.clone()))
    }
}
