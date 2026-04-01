// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::fixed_size_list::vtable::FixedSizeList;
use crate::arrays::fixed_size_list::vtable::FixedSizeListArray;
use crate::validity::Validity;
use crate::vtable::ValidityVTable;

impl ValidityVTable<FixedSizeList> for FixedSizeList {
    fn validity(array: &FixedSizeListArray) -> VortexResult<Validity> {
        Ok(array.validity())
    }
}
