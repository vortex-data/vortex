// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::{Itertools, MinMaxResult};
use vortex_dtype::DecimalDType;
use vortex_error::VortexExpect;
use vortex_scalar::{DecimalType, i256};

use crate::arrays::DecimalArray;
use crate::vtable::ValidityHelper;

/// Maps a decimal precision into the smallest type that can represent it.
pub fn smallest_decimal_value_type(decimal_dtype: &DecimalDType) -> DecimalType {
    match decimal_dtype.precision() {
        1..=2 => DecimalType::I8,
        3..=4 => DecimalType::I16,
        5..=9 => DecimalType::I32,
        10..=18 => DecimalType::I64,
        19..=38 => DecimalType::I128,
        39..=76 => DecimalType::I256,
        0 => unreachable!("precision must be greater than 0"),
        p => unreachable!("precision larger than 76 is invalid found precision {p}"),
    }
}

/// True if `value_type` can represent every value of the type `dtype`.
pub fn is_compatible_decimal_value_type(value_type: DecimalType, dtype: DecimalDType) -> bool {
    value_type >= smallest_decimal_value_type(&dtype)
}

macro_rules! try_downcast {
    ($array:expr, from: $src:ty, to: $($dst:ty),*) => {{
        use vortex_dtype::BigCast;

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
