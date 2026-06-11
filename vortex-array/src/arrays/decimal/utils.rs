// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use itertools::MinMaxResult;
use vortex_error::VortexExpect;

use crate::arrays::DecimalArray;
use crate::dtype::BigCast;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::i256;

/// Compute the `(min, max)` of the array's values, interpreted as `T`.
///
/// Returns `None` when the array has fewer than two elements, in which case there is nothing to
/// narrow.
fn min_max<T: NativeDecimalType>(array: &DecimalArray) -> Option<(T, T)> {
    match array.buffer::<T>().iter().copied().minmax() {
        MinMaxResult::MinMax(min, max) => Some((min, max)),
        MinMaxResult::NoElements | MinMaxResult::OneElement(_) => None,
    }
}

/// Attempt to narrow `array` (holding values of type `Src`) to the smaller `Dst` type.
///
/// Returns the narrowed array if both `min` and `max` are representable in `Dst`, otherwise `None`.
fn try_downcast<Src, Dst>(array: &DecimalArray, min: Src, max: Src) -> Option<DecimalArray>
where
    Src: NativeDecimalType,
    Dst: NativeDecimalType,
{
    (<Dst as BigCast>::from(min).is_some() && <Dst as BigCast>::from(max).is_some()).then(|| {
        DecimalArray::new::<Dst>(
            array
                .buffer::<Src>()
                .into_iter()
                .map(|v| <Dst as BigCast>::from(v).vortex_expect("decimal conversion failure"))
                .collect(),
            array.decimal_dtype(),
            array
                .validity()
                .vortex_expect("decimal validity should be derivable"),
        )
    })
}

/// Attempt to narrow the decimal array to any smaller supported type.
pub fn narrowed_decimal(decimal_array: DecimalArray) -> DecimalArray {
    match decimal_array.values_type() {
        // Cannot narrow any more
        DecimalType::I8 => decimal_array,
        DecimalType::I16 => match min_max::<i16>(&decimal_array) {
            None => decimal_array,
            Some((min, max)) => {
                try_downcast::<i16, i8>(&decimal_array, min, max).unwrap_or(decimal_array)
            }
        },
        DecimalType::I32 => match min_max::<i32>(&decimal_array) {
            None => decimal_array,
            Some((min, max)) => try_downcast::<i32, i8>(&decimal_array, min, max)
                .or_else(|| try_downcast::<i32, i16>(&decimal_array, min, max))
                .unwrap_or(decimal_array),
        },
        DecimalType::I64 => match min_max::<i64>(&decimal_array) {
            None => decimal_array,
            Some((min, max)) => try_downcast::<i64, i8>(&decimal_array, min, max)
                .or_else(|| try_downcast::<i64, i16>(&decimal_array, min, max))
                .or_else(|| try_downcast::<i64, i32>(&decimal_array, min, max))
                .unwrap_or(decimal_array),
        },
        DecimalType::I128 => match min_max::<i128>(&decimal_array) {
            None => decimal_array,
            Some((min, max)) => try_downcast::<i128, i8>(&decimal_array, min, max)
                .or_else(|| try_downcast::<i128, i16>(&decimal_array, min, max))
                .or_else(|| try_downcast::<i128, i32>(&decimal_array, min, max))
                .or_else(|| try_downcast::<i128, i64>(&decimal_array, min, max))
                .unwrap_or(decimal_array),
        },
        DecimalType::I256 => match min_max::<i256>(&decimal_array) {
            None => decimal_array,
            Some((min, max)) => try_downcast::<i256, i8>(&decimal_array, min, max)
                .or_else(|| try_downcast::<i256, i16>(&decimal_array, min, max))
                .or_else(|| try_downcast::<i256, i32>(&decimal_array, min, max))
                .or_else(|| try_downcast::<i256, i64>(&decimal_array, min, max))
                .or_else(|| try_downcast::<i256, i128>(&decimal_array, min, max))
                .unwrap_or(decimal_array),
        },
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use super::*;
    use crate::dtype::DecimalDType;
    use crate::validity::Validity;

    #[test]
    fn narrows_to_smallest_fitting_type() {
        // i32 values that fit within i8 should narrow all the way to i8.
        let array = DecimalArray::new(
            buffer![1i32, -5i32, 100i32],
            DecimalDType::new(9, 2),
            Validity::NonNullable,
        );
        assert_eq!(narrowed_decimal(array).values_type(), DecimalType::I8);
    }

    #[test]
    fn narrows_to_intermediate_type() {
        // i64 values that exceed i8 but fit within i16.
        let array = DecimalArray::new(
            buffer![1i64, -5000i64, 30000i64],
            DecimalDType::new(18, 2),
            Validity::NonNullable,
        );
        assert_eq!(narrowed_decimal(array).values_type(), DecimalType::I16);
    }

    #[test]
    fn keeps_type_when_values_do_not_fit() {
        // Value too large for any smaller type than i64.
        let array = DecimalArray::new(
            buffer![1i64, i64::MAX],
            DecimalDType::new(18, 2),
            Validity::NonNullable,
        );
        assert_eq!(narrowed_decimal(array).values_type(), DecimalType::I64);
    }

    #[test]
    fn keeps_type_with_fewer_than_two_elements() {
        // A single-element array carries no range to narrow against.
        let array = DecimalArray::new(
            buffer![1i32],
            DecimalDType::new(9, 2),
            Validity::NonNullable,
        );
        assert_eq!(narrowed_decimal(array).values_type(), DecimalType::I32);
    }
}
