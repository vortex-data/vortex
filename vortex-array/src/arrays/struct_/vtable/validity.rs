// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::struct_::vtable::Struct;
use crate::validity::Validity;
use crate::vtable::ArrayView;
use crate::vtable::ValidityVTable;

impl ValidityVTable<Struct> for Struct {
    fn validity(array: ArrayView<'_, Struct>) -> VortexResult<Validity> {
        Ok(array.data().validity())
    }
}
