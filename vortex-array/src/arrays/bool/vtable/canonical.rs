// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Canonical;
use crate::arrays::BoolArray;
use crate::arrays::BoolVTable;
use crate::builders::ArrayBuilder;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<BoolVTable> for BoolVTable {
    fn canonicalize(array: &BoolArray) -> Canonical {
        Canonical::Bool(array.clone())
    }

    fn append_to_builder(array: &BoolArray, builder: &mut dyn ArrayBuilder) {
        builder.extend_from_array(array.as_ref())
    }
}
