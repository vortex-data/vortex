// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::primitive::vtable::Primitive;
use crate::validity::Validity;

impl ValidityVTable<Primitive> for Primitive {
    fn validity(array: ArrayView<'_, Primitive>) -> VortexResult<Validity> {
        Ok(array.data().validity())
    }
}
