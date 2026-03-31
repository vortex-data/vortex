// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::StructArray;
use crate::arrays::struct_::vtable::Struct;
use crate::validity::Validity;
use crate::vtable::ValidityVTable;

impl ValidityVTable<Struct> for Struct {
    fn validity(array: &StructArray) -> VortexResult<Validity> {
        Ok(array.validity())
    }
}
