// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Derive decimal validity from the integer child.
//!
//! The canonical decimal has no separate validity slot. Delegating to the values child keeps
//! its nullability and validity aligned with the decimal dtype.

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
