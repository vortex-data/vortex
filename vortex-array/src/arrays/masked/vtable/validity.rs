// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::masked::vtable::Masked;
use crate::arrays::masked::MaskedArrayExt;
use crate::validity::Validity;

impl ValidityVTable<Masked> for Masked {
    fn validity(array: ArrayView<'_, Masked>) -> VortexResult<Validity> {
        Ok(array.masked_validity())
    }
}
