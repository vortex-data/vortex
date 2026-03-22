// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;

use crate::arrays::DecimalArray;
use crate::match_each_decimal_value_type;

pub(super) fn check_decimal_constant(array: &DecimalArray) -> bool {
    match_each_decimal_value_type!(array.values_type(), |S| {
        array.buffer::<S>().iter().all_equal()
    })
}
