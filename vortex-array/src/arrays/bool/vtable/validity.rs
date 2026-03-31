// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::bool::vtable::BoolArray;
use crate::arrays::bool::vtable::Bool;
use crate::validity::Validity;
use crate::vtable::ValidityVTable;

impl ValidityVTable<Bool> for Bool {
    fn validity(array: &BoolArray) -> VortexResult<Validity> {
        Ok(array.validity())
    }
}
