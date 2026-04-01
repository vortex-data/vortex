// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::primitive::vtable::Primitive;
use crate::validity::Validity;
use crate::vtable::ArrayView;
use crate::vtable::ValidityVTable;

impl ValidityVTable<Primitive> for Primitive {
    fn validity(array: ArrayView<'_, Primitive>) -> VortexResult<Validity> {
        Ok(array.data().validity())
    }
}
