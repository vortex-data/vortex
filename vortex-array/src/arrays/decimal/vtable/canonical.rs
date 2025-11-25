// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Canonical;
use crate::arrays::DecimalArray;
use crate::arrays::DecimalVTable;
use crate::builders::ArrayBuilder;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<DecimalVTable> for DecimalVTable {
    fn canonicalize(array: &DecimalArray) -> Canonical {
        Canonical::Decimal(array.clone())
    }

    fn append_to_builder(array: &DecimalArray, builder: &mut dyn ArrayBuilder) {
        builder.extend_from_array(array.as_ref())
    }
}
