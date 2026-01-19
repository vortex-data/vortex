// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::validity::Validity;
use crate::vtable::ValidityVTable;

impl ValidityVTable<ConstantVTable> for ConstantVTable {
    fn validity(array: &ConstantArray) -> VortexResult<Validity> {
        Ok(if array.scalar().is_null() {
            Validity::AllInvalid
        } else {
            Validity::AllValid
        })
    }

    fn validity_mask(array: &ConstantArray) -> Mask {
        match array.scalar().is_null() {
            true => Mask::AllFalse(array.len),
            false => Mask::AllTrue(array.len),
        }
    }
}
