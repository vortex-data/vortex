// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use vortex_array::arrays::DecimalArray;
use vortex_array::match_each_decimal_value_type;

pub(super) fn check_decimal_constant(array: &DecimalArray) -> bool {
    match_each_decimal_value_type!(array.values_type(), |S| {
        values_are_constant(array.buffer::<S>().as_slice())
    })
}

fn values_are_constant<T: Eq>(values: &[T]) -> bool {
    values.iter().all_equal()
}
