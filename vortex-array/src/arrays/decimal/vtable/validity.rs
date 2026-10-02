// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::ArrayRef;
use crate::array::ArrayView;
use crate::array::ValidityChild;
use crate::arrays::Decimal;
use crate::arrays::decimal::DecimalArraySlotsExt;

impl ValidityChild<Decimal> for Decimal {
    fn validity_child(array: ArrayView<'_, Decimal>) -> ArrayRef {
        array.values().clone()
    }
}
