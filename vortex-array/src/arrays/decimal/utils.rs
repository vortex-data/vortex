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

/// Build the `narrowed_decimal` match from a single ordered list of native decimal types.
///
/// The native types form a total order (each is losslessly representable in the next), so for any
/// type `T` the set of "smaller" types is exactly the prefix of the list before `T`. The macro
/// walks the list left to right, accumulating that prefix, and for each type emits a match arm that
/// tries [`try_downcast`] against every smaller type, smallest first. The first arm (no smaller
/// types) cannot narrow and is returned as-is.
macro_rules! narrow_to_smallest {
    // Entry point: seed the walk with an empty prefix, the full ordered list, and no arms yet.
    ($array:ident) => {
        narrow_to_smallest!(@build $array; smaller: [];
            rest: [(i8, I8), (i16, I16), (i32, I32), (i64, I64), (i128, I128), (i256, I256)];
            arms: [])
    };

    // List exhausted: emit the assembled match.
    (@build $array:ident; smaller: $smaller:tt; rest: []; arms: [$($arm:tt)*]) => {
        match $array.values_type() {
            $($arm)*
        }
    };

    // Head type has no smaller types: it cannot be narrowed.
    (@build $array:ident; smaller: []; rest: [($t:ty, $variant:ident) $(, $rest:tt)*]; arms: [$($arm:tt)*]) => {
        narrow_to_smallest!(@build $array; smaller: [$t]; rest: [$($rest),*];
            arms: [$($arm)* DecimalType::$variant => $array,])
    };

    // Head type with a non-empty prefix of smaller types: try each, smallest first.
    (@build $array:ident; smaller: [$($smaller:ty),+]; rest: [($t:ty, $variant:ident) $(, $rest:tt)*]; arms: [$($arm:tt)*]) => {
        narrow_to_smallest!(@build $array; smaller: [$($smaller,)+ $t]; rest: [$($rest),*];
            arms: [$($arm)*
                DecimalType::$variant => match min_max::<$t>(&$array) {
                    None => $array,
                    Some((min, max)) => Option::<DecimalArray>::None
                        $(.or_else(|| try_downcast::<$t, $smaller>(&$array, min, max)))+
                        .unwrap_or($array),
                },
            ])
    };
}

/// Attempt to narrow the decimal array to the smallest supported type that fits its values.
pub fn narrowed_decimal(decimal_array: DecimalArray) -> DecimalArray {
    narrow_to_smallest!(decimal_array)
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
