// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_mask::Mask;

use crate::arrays::{ConstantArray, ConstantVTable};
use crate::vtable::ValidityVTable;

impl ValidityVTable<ConstantVTable> for ConstantVTable {
    fn is_valid(array: &ConstantArray, _index: usize) -> bool {
        !array.scalar().is_null()
    }

    fn all_valid(array: &ConstantArray) -> bool {
        !array.scalar().is_null()
    }

    fn all_invalid(array: &ConstantArray) -> bool {
        array.scalar().is_null()
    }

    fn validity_mask(array: &ConstantArray) -> Mask {
        match array.scalar().is_null() {
            true => Mask::AllFalse(array.len),
            false => Mask::AllTrue(array.len),
        }
    }
}
