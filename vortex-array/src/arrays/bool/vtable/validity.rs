// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::bool::vtable::Bool;
use crate::validity::Validity;

impl ValidityVTable<Bool> for Bool {
    fn validity(array: ArrayView<'_, Bool>) -> VortexResult<Validity> {
        Ok(array.data().validity())
    }
}
