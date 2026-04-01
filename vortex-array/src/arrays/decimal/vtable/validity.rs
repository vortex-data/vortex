// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::decimal::vtable::Decimal;
use crate::validity::Validity;
use crate::vtable::ArrayView;
use crate::vtable::ValidityVTable;

impl ValidityVTable<Decimal> for Decimal {
    fn validity(array: ArrayView<'_, Decimal>) -> VortexResult<Validity> {
        Ok(array.data().validity())
    }
}
