// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::arrays::DecimalArray;
use crate::match_each_decimal_value_type;

pub(super) fn check_decimal_identical(
    lhs: &DecimalArray,
    rhs: &DecimalArray,
) -> VortexResult<bool> {
    Ok(match_each_decimal_value_type!(lhs.values_type(), |T| {
        decimal_values_equal(lhs.buffer::<T>().as_slice(), rhs.buffer::<T>().as_slice())
    }))
}

fn decimal_values_equal<T: PartialEq>(lhs: &[T], rhs: &[T]) -> bool {
    lhs == rhs
}
