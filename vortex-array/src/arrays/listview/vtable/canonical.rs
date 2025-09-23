// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Canonical;
use crate::arrays::{ListViewArray, ListViewVTable};
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<ListViewVTable> for ListViewVTable {
    fn canonicalize(array: &ListViewArray) -> Canonical {
        Canonical::List(array.clone())
    }
}
