// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::list::vtable::List;
use crate::validity::Validity;
use crate::vtable::ArrayView;
use crate::vtable::ValidityVTable;

impl ValidityVTable<List> for List {
    fn validity(array: ArrayView<'_, List>) -> VortexResult<Validity> {
        Ok(array.data().validity())
    }
}
