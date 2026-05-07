// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::Variant;
use crate::arrays::variant::VariantArrayExt;
use crate::validity::Validity;

impl ValidityVTable<Variant> for Variant {
    fn validity(array: ArrayView<'_, Variant>) -> VortexResult<Validity> {
        array.core_storage().validity()
    }
}
