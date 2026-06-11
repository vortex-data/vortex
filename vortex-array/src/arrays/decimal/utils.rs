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

/// Compute the `(min, max)` of the array's values, widened to [`i256`].
///
/// The values are read as `Src` (their physical storage type) and each bound is upcast to `i256`,
/// the widest decimal type, so the range can be examined independently of the storage type.
/// Returns `None` when the array has fewer than two elements, in which case there is nothing to
/// narrow.
fn upcast_minmax<Src: NativeDecimalType>(array: &DecimalArray) -> Option<(i256, i256)> {
    match array.buffer::<Src>().iter().copied().minmax() {
        MinMaxResult::MinMax(min, max) => Some((
            min.to_i256()
                .vortex_expect("native decimal value fits in i256"),
            max.to_i256()
                .vortex_expect("native decimal value fits in i256"),
        )),
        MinMaxResult::NoElements | MinMaxResult::OneElement(_) => None,
    }
}

/// Find the smallest decimal type whose value range contains both `min` and `max`.
///
/// The native types form a total order (`i8` ⊂ `i16` ⊂ ... ⊂ `i256`), so this scans from smallest
/// to largest and returns the first that fits. Since every value already fits its own storage type,
/// the result is never wider than the array's current type.
fn smallest_fitting_type(min: i256, max: i256) -> DecimalType {
    fn fits<T: BigCast>(min: i256, max: i256) -> bool {
        <T as BigCast>::from(min).is_some() && <T as BigCast>::from(max).is_some()
    }

    if fits::<i8>(min, max) {
        DecimalType::I8
    } else if fits::<i16>(min, max) {
        DecimalType::I16
    } else if fits::<i32>(min, max) {
        DecimalType::I32
    } else if fits::<i64>(min, max) {
        DecimalType::I64
    } else if fits::<i128>(min, max) {
        DecimalType::I128
    } else {
        DecimalType::I256
    }
}

/// Infallibly cast every value of `array` from `Src` to the narrower `Dst`.
///
/// The caller must have established that all values fit in `Dst` (see [`smallest_fitting_type`]),
/// so the per-element conversion cannot fail.
fn cast_values<Src: NativeDecimalType, Dst: NativeDecimalType>(
    array: &DecimalArray,
) -> DecimalArray {
    DecimalArray::new::<Dst>(
        array
            .buffer::<Src>()
            .into_iter()
            .map(|v| <Dst as BigCast>::from(v).vortex_expect("value fits the chosen decimal type"))
            .collect(),
        array.decimal_dtype(),
        array
            .validity()
            .vortex_expect("decimal validity should be derivable"),
    )
}

/// Attempt to narrow the decimal array to the smallest supported type that fits its values.
///
/// First the value range is computed as [`i256`], then the smallest fitting type is chosen, and
/// finally the array is cast to that type via an infallible double dispatch over the source and
/// target types.
pub fn narrowed_decimal(decimal_array: DecimalArray) -> DecimalArray {
    // Step 1: compute the value range, widened to i256, dispatching on the storage type.
    let minmax = match decimal_array.values_type() {
        DecimalType::I8 => upcast_minmax::<i8>(&decimal_array),
        DecimalType::I16 => upcast_minmax::<i16>(&decimal_array),
        DecimalType::I32 => upcast_minmax::<i32>(&decimal_array),
        DecimalType::I64 => upcast_minmax::<i64>(&decimal_array),
        DecimalType::I128 => upcast_minmax::<i128>(&decimal_array),
        DecimalType::I256 => upcast_minmax::<i256>(&decimal_array),
    };
    let Some((min, max)) = minmax else {
        return decimal_array;
    };

    // Step 2: pick the smallest type that can hold the whole range.
    let target = smallest_fitting_type(min, max);

    // Step 3: infallibly cast to the target. The outer match selects the source type, the inner
    // match selects the (smaller) target type; equal-or-wider targets need no work.
    match decimal_array.values_type() {
        DecimalType::I8 => decimal_array,
        DecimalType::I16 => match target {
            DecimalType::I8 => cast_values::<i16, i8>(&decimal_array),
            _ => decimal_array,
        },
        DecimalType::I32 => match target {
            DecimalType::I8 => cast_values::<i32, i8>(&decimal_array),
            DecimalType::I16 => cast_values::<i32, i16>(&decimal_array),
            _ => decimal_array,
        },
        DecimalType::I64 => match target {
            DecimalType::I8 => cast_values::<i64, i8>(&decimal_array),
            DecimalType::I16 => cast_values::<i64, i16>(&decimal_array),
            DecimalType::I32 => cast_values::<i64, i32>(&decimal_array),
            _ => decimal_array,
        },
        DecimalType::I128 => match target {
            DecimalType::I8 => cast_values::<i128, i8>(&decimal_array),
            DecimalType::I16 => cast_values::<i128, i16>(&decimal_array),
            DecimalType::I32 => cast_values::<i128, i32>(&decimal_array),
            DecimalType::I64 => cast_values::<i128, i64>(&decimal_array),
            _ => decimal_array,
        },
        DecimalType::I256 => match target {
            DecimalType::I8 => cast_values::<i256, i8>(&decimal_array),
            DecimalType::I16 => cast_values::<i256, i16>(&decimal_array),
            DecimalType::I32 => cast_values::<i256, i32>(&decimal_array),
            DecimalType::I64 => cast_values::<i256, i64>(&decimal_array),
            DecimalType::I128 => cast_values::<i256, i128>(&decimal_array),
            _ => decimal_array,
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

    #[test]
    fn narrows_from_widest_type() {
        // i256 values that fit within i32 (but not i16) should land on i32, exercising the i256
        // upcast and the widest source dispatch arm.
        let array = DecimalArray::new(
            buffer![i256::from_i128(-100_000), i256::from_i128(100_000)],
            DecimalDType::new(76, 2),
            Validity::NonNullable,
        );
        assert_eq!(narrowed_decimal(array).values_type(), DecimalType::I32);
    }
}
