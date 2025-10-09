// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Canonical;
use crate::arrays::struct_::{
    StructArray,
    StructVTable,
};
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<StructVTable> for StructVTable {
    fn canonicalize(array: &StructArray) -> Canonical {
        Canonical::Struct(array.clone())
    }
}
