// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::list::vtable::List;
use crate::validity::Validity;

impl ValidityVTable<List> for List {
    fn validity(array: ArrayView<'_, List>) -> VortexResult<Validity> {
        Ok(array.data().validity())
    }
}
