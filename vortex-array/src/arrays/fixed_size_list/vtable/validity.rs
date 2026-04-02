// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::fixed_size_list::vtable::FixedSizeList;
use crate::validity::Validity;

impl ValidityVTable<FixedSizeList> for FixedSizeList {
    fn validity(array: ArrayView<'_, FixedSizeList>) -> VortexResult<Validity> {
        Ok(array.data().validity())
    }
}
