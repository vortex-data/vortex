// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::arrays::masked::MaskedArray;
use crate::arrays::MaskedVTable;
use crate::vtable::ValidityVTable;
use crate::Array;
use std::ops::BitAnd;
use vortex_mask::Mask;

impl ValidityVTable<MaskedVTable> for MaskedVTable {
    fn is_valid(array: &MaskedArray, index: usize) -> bool {
        array.child.is_valid(index) && array.validity.is_valid(index)
    }

    fn all_valid(array: &MaskedArray) -> bool {
        array.child.all_valid() && array.validity.all_valid(array.len())
    }

    fn all_invalid(array: &MaskedArray) -> bool {
        array.child.all_invalid() || array.validity.all_invalid(array.len())
    }

    fn validity_mask(array: &MaskedArray) -> Mask {
        let child_mask = array.child.validity_mask();
        let own_mask = array.validity.to_mask(array.len());
        child_mask.bitand(&own_mask)
    }
}
