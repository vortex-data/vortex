// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::Canonical;
use crate::arrays::ListViewArray;
use crate::arrays::ListViewVTable;
use crate::vtable::CanonicalVTable;

impl CanonicalVTable<ListViewVTable> for ListViewVTable {
    fn canonicalize(array: &ListViewArray) -> VortexResult<Canonical> {
        Ok(Canonical::List(array.clone()))
    }
}
