// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityVTable;
use vortex_error::VortexResult;

use crate::BitPacked;
use crate::BitPackedArray;

impl ValidityVTable<BitPacked> for BitPacked {
    fn validity(array: &BitPackedArray) -> VortexResult<Validity> {
        Ok(array.validity())
    }
}
