// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use itertools::MinMaxResult;
use vortex_error::VortexExpect;

use crate::arrays::DecimalArray;
use crate::dtype::DecimalType;
use crate::dtype::i256;

macro_rules! try_downcast {
    ($array:expr, from: $src:ty, to: $($dst:ty),*) => {{
        use crate::dtype::BigCast;

        // Collect the min/max of the values
        let minmax = $array.buffer::<$src>().iter().copied().minmax();
        match minmax {
            MinMaxResult::NoElements => return $array,
            MinMaxResult::OneElement(_) => return $array,
            MinMaxResult::MinMax(min, max) => {
                $(
                    if <$dst as BigCast>::from(min).is_some() && <$dst as BigCast>::from(max).is_some() {
                        return DecimalArray::new::<$dst>(
                            $array
                                .buffer::<$src>()
                                .into_iter()
                                .map(|v| <$dst as BigCast>::from(v).vortex_expect("decimal conversion failure"))
                                .collect(),
                            $array.decimal_dtype(),
                            $array.validity().clone(),
                        );
                    }
                )*

                return $array;
            }
        }
    }};
}

/// Attempt to narrow the decimal array to any smaller supported type.
pub fn narrowed_decimal(decimal_array: DecimalArray) -> DecimalArray {
    match decimal_array.values_type() {
        // Cannot narrow any more
        DecimalType::I8 => decimal_array,
        DecimalType::I16 => {
            try_downcast!(decimal_array, from: i16, to: i8)
        }
        DecimalType::I32 => {
            try_downcast!(decimal_array, from: i32, to: i8, i16)
        }
        DecimalType::I64 => {
            try_downcast!(decimal_array, from: i64, to: i8, i16, i32)
        }
        DecimalType::I128 => {
            try_downcast!(decimal_array, from: i128, to: i8, i16, i32, i64)
        }
        DecimalType::I256 => {
            try_downcast!(decimal_array, from: i256, to: i8, i16, i32, i64, i128)
        }
    }
}
