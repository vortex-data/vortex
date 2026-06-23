// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use itertools::MinMaxResult;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::arrays::DecimalArray;
use crate::dtype::BigCast;
use crate::dtype::DecimalType;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;

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
                            $array
                                .validity()
                                .vortex_expect("decimal validity should be derivable"),
                        );
                    }
                )*

                return $array;
            }
        }
    }};
}

/// Cast the array's physical values to `target`, preserving the logical decimal dtype and
/// validity.
///
/// `mask` must be the materialized validity of `array`. Null slots are unconstrained by the
/// [`DecimalArray`] invariants (only *non-null* values must fit the precision) and may hold
/// bytes that do not fit `target`, so they are replaced with zero rather than cast.
///
/// # Errors
///
/// Returns an error if a non-null value cannot be represented in `target`.
pub fn cast_decimal_values(
    array: &DecimalArray,
    target: DecimalType,
    mask: &Mask,
) -> VortexResult<DecimalArray> {
    let decimal_dtype = array.decimal_dtype();
    let validity = array.validity()?;
    match_each_decimal_value_type!(array.values_type(), |F| {
        let from = array.buffer::<F>();
        match_each_decimal_value_type!(target, |T| {
            let values = from
                .iter()
                .enumerate()
                .map(|(i, &v)| {
                    if mask.value(i) {
                        <T as BigCast>::from(v).ok_or_else(|| {
                            vortex_err!("decimal value {v} does not fit values type {target}")
                        })
                    } else {
                        Ok(T::default())
                    }
                })
                .collect::<VortexResult<Buffer<T>>>()?;
            Ok(DecimalArray::new::<T>(values, decimal_dtype, validity))
        })
    })
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

#[cfg(test)]
mod tests {
    use vortex_buffer::BitBuffer;
    use vortex_buffer::Buffer;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::cast_decimal_values;
    use crate::arrays::DecimalArray;
    use crate::dtype::DecimalDType;
    use crate::dtype::DecimalType;
    use crate::validity::Validity;

    #[test]
    fn cast_zeroes_garbage_null_slots() -> VortexResult<()> {
        let dt = DecimalDType::new(5, 2);
        let validity = Validity::from(BitBuffer::from_iter([true, false, true]));
        let arr = DecimalArray::new::<i64>(
            Buffer::<i64>::copy_from([7i64, i64::MAX, -99_999]),
            dt,
            validity,
        );
        let mask = Mask::from_iter([true, false, true]);
        let narrowed = cast_decimal_values(&arr, DecimalType::I32, &mask)?;
        assert_eq!(narrowed.values_type(), DecimalType::I32);
        assert_eq!(narrowed.buffer::<i32>().as_slice(), &[7, 0, -99_999]);
        Ok(())
    }

    #[test]
    fn cast_rejects_non_null_value_that_does_not_fit() {
        let dt = DecimalDType::new(5, 2);
        let arr = DecimalArray::new::<i64>(
            Buffer::<i64>::copy_from([i64::MAX]),
            dt,
            Validity::NonNullable,
        );
        let mask = Mask::new_true(1);
        assert!(cast_decimal_values(&arr, DecimalType::I32, &mask).is_err());
    }
}
