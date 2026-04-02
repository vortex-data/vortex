// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::decimal::vtable::Decimal;
use crate::arrays::decimal::vtable::DecimalArray;
use crate::validity::Validity;
use crate::vtable::ValidityVTable;

impl ValidityVTable<Decimal> for Decimal {
    fn validity(array: &DecimalArray) -> VortexResult<Validity> {
        Ok(array.validity())
    }
}
