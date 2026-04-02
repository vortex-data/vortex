// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::listview::vtable::ListView;
use crate::validity::Validity;

impl ValidityVTable<ListView> for ListView {
    fn validity(array: ArrayView<'_, ListView>) -> VortexResult<Validity> {
        Ok(array.data().validity())
    }
}
