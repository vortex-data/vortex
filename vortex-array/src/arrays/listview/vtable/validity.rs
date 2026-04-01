// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::listview::vtable::ListView;
use crate::validity::Validity;
use crate::vtable::ArrayView;
use crate::vtable::ValidityVTable;

impl ValidityVTable<ListView> for ListView {
    fn validity(array: ArrayView<'_, ListView>) -> VortexResult<Validity> {
        Ok(array.data().validity())
    }
}
