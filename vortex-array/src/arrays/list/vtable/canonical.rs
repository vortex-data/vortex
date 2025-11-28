// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Canonical;
use crate::arrays::ListArray;
use crate::arrays::ListVTable;
use crate::arrays::list_view_from_list;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<ListVTable> for ListVTable {
    fn canonicalize(array: &ListArray) -> VortexResult<Canonical> {
        Ok(Canonical::List(list_view_from_list(array.clone())))
    }
}
