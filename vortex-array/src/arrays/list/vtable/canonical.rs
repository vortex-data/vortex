// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Canonical;
use crate::arrays::{ListArray, ListVTable, list_view_from_list};
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<ListVTable> for ListVTable {
    fn canonicalize(array: &ListArray) -> Canonical {
        Canonical::List(list_view_from_list(array.clone()))
    }
}
