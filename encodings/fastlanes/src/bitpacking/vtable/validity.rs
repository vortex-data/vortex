// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityVTable;
use vortex_error::VortexResult;

use crate::BitPacked;

impl ValidityVTable<BitPacked> for BitPacked {
    fn validity(array: ArrayView<'_, BitPacked>) -> VortexResult<Validity> {
        Ok(array.data().validity())
    }
}
