// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Canonical;
use crate::arrays::PrimitiveArray;
use crate::arrays::PrimitiveVTable;
use crate::builders::ArrayBuilder;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<PrimitiveVTable> for PrimitiveVTable {
    fn canonicalize(array: &PrimitiveArray) -> Canonical {
        Canonical::Primitive(array.clone())
    }

    fn append_to_builder(array: &PrimitiveArray, builder: &mut dyn ArrayBuilder) {
        builder.extend_from_array(array.as_ref())
    }
}
