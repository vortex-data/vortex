// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::masked::MaskedArrayExt;
use crate::arrays::masked::MaskedArraySlotsExt;
use crate::arrays::masked::vtable::Masked;
use crate::validity::Validity;

impl ValidityVTable<Masked> for Masked {
    fn validity(array: ArrayView<'_, Masked>) -> VortexResult<Validity> {
        // The child may carry its own nulls; lazily merge them with the mask.
        array.child().validity()?.and(array.masked_validity())
    }
}
