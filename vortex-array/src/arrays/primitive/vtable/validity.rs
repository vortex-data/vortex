// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::PrimitiveArray;
use crate::arrays::primitive::vtable::Primitive;
use crate::validity::Validity;
use crate::vtable::ValidityVTable;

impl ValidityVTable<Primitive> for Primitive {
    fn validity(array: &PrimitiveArray) -> VortexResult<Validity> {
        Ok(array.validity())
    }
}
