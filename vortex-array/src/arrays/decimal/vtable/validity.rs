// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::array::ArrayView;
use crate::array::ValidityVTable;
use crate::arrays::decimal::vtable::Decimal;
use crate::validity::Validity;

impl ValidityVTable<Decimal> for Decimal {
    fn validity(array: ArrayView<'_, Decimal>) -> VortexResult<Validity> {
        Ok(array.data().validity())
    }
}
