// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::ConstantArray;
use crate::arrays::ConstantVTable;
use crate::validity::Validity;
use crate::vtable::ValidityVTable;

impl ValidityVTable<ConstantVTable> for ConstantVTable {
    fn validity(array: &ConstantArray) -> VortexResult<Validity> {
        debug_assert!(array.dtype().is_nullable());
        Ok(if array.scalar().is_null() {
            Validity::AllInvalid
        } else {
            Validity::AllValid
        })
    }
}
