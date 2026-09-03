// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::ArrayView;
use vortex_array::validity::Validity;
use vortex_array::vtable::ValidityVTable;
use vortex_error::VortexResult;

use crate::BitPackedV2;
use crate::BitPackedV2ArrayExt;

impl ValidityVTable<BitPackedV2> for BitPackedV2 {
    fn validity(array: ArrayView<'_, BitPackedV2>) -> VortexResult<Validity> {
        Ok(BitPackedV2ArrayExt::validity(&array))
    }
}
