// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::FixedSizeBinary;
use crate::arrays::fixed_size_binary::FixedSizeBinaryArrayExt;
use crate::validity::Validity;

impl ValidityVTable<FixedSizeBinary> for FixedSizeBinary {
    fn validity(array: ArrayView<'_, FixedSizeBinary>) -> VortexResult<Validity> {
        Ok(FixedSizeBinaryArrayExt::validity(&array))
    }
}
