// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::masked::vtable::Masked;
use crate::validity::Validity;
use crate::vtable::ArrayView;
use crate::vtable::ValidityVTable;

impl ValidityVTable<Masked> for Masked {
    fn validity(array: ArrayView<'_, Masked>) -> VortexResult<Validity> {
        Ok(array.data().validity())
    }
}
