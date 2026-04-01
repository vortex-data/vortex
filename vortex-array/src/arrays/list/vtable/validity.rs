// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::List;
use crate::arrays::list::vtable::ListArray;
use crate::validity::Validity;
use crate::vtable::ValidityVTable;

impl ValidityVTable<List> for List {
    fn validity(array: &ListArray) -> VortexResult<Validity> {
        Ok(array.validity())
    }
}
