// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use itertools::MinMaxResult;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_compute::lane_kernels::IndexedSourceExt;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_ensure_eq;
use vortex_mask::AllOr;
use vortex_mask::Mask;

use crate::arrays::DecimalArray;
use crate::arrays::decimal::DecimalArrayExt;
use crate::buffer::BufferHandle;
use crate::dtype::BigCast;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::i256;
use crate::match_each_decimal_value_type;

/// Return the array's unscaled values widened to `W`, which must be at least as wide as the
/// array's storage type.
pub(crate) fn widened_buffer<W: NativeDecimalType>(array: &DecimalArray) -> Buffer<W> {
    if array.values_type() == W::DECIMAL_TYPE {
        return array.buffer::<W>();
    }
    match_each_decimal_value_type!(array.values_type(), |T| {
        // Only storage types narrower than `W` reach here. The const condition lets the compiler
        // skip codegen of the other arms.
        if const { (T::DECIMAL_TYPE as u8) < (W::DECIMAL_TYPE as u8) } {
            array
                .buffer::<T>()
                .iter()
                .map(|v| W::from(*v).vortex_expect("widening decimal cast must succeed"))
                .collect()
        } else {
            unreachable!(
                "{} is not narrower than {}",
                T::DECIMAL_TYPE,
                W::DECIMAL_TYPE
            )
        }
    })
}

/// Return the array's unscaled values converted to exactly `W`, whatever the array's
/// storage type. Zero-copy when the array is already stored at `W`.
///
/// Widening is lossless. Narrowing fails for any *valid* value that does not fit `W`.
/// Null slots may hold arbitrary bytes and never fail; their contents in the returned
/// buffer are likewise arbitrary and must not be read.
pub fn converted_buffer<W: NativeDecimalType>(
    array: &DecimalArray,
    validity: &Mask,
) -> VortexResult<Buffer<W>> {
    // Widening can never fail, so it needs no validation pass.
    if array.values_type() <= W::DECIMAL_TYPE {
        return Ok(widened_buffer(array));
    }
    match_each_decimal_value_type!(array.values_type(), |T| {
        // Only storage types wider than `W` reach here. The const condition lets the compiler
        // skip codegen of the other arms.
        if const { (T::DECIMAL_TYPE as u8) > (W::DECIMAL_TYPE as u8) } {
            let src = array.buffer::<T>();
            match validity {
                Mask::AllTrue(_) => {
                    // Keeping the overflow scan branchless and vectorizable. Only on overflow
                    // do we rescan for diagnostics.
                    let any_overflow = src.iter().fold(false, |acc, v| acc | W::from(*v).is_none());
                    if any_overflow {
                        let (i, v) = src
                            .iter()
                            .enumerate()
                            .find(|&(_, v)| W::from(*v).is_none())
                            .vortex_expect("overflow scan found an overflowing value");
                        vortex_bail!(
                            "decimal value {v} at index {i} does not fit {}",
                            W::DECIMAL_TYPE
                        );
                    }
                }
                Mask::AllFalse(_) => return Ok(Buffer::zeroed(src.len())),
                Mask::Values(values) => {
                    let any_overflow = src.iter().fold(false, |acc, v| acc | W::from(*v).is_none());
                    if any_overflow {
                        for (i, v) in src.iter().enumerate() {
                            if values.value(i) && W::from(*v).is_none() {
                                vortex_bail!(
                                    "decimal value {v} at index {i} does not fit {}",
                                    W::DECIMAL_TYPE
                                );
                            }
                        }
                    }
                }
            }
            // The convert pass is infallible: every valid value fits, and out-of-range null-slot
            // garbage is mapped to the default value. Callers must still ignore null slots.
            Ok(src
                .iter()
                .map(|v| W::from(*v).unwrap_or_default())
                .collect())
        } else {
            unreachable!("{} is not wider than {}", T::DECIMAL_TYPE, W::DECIMAL_TYPE)
        }
    })
}

/// Copies unscaled decimal values into a new buffer at the canonical width of `target`.
///
/// The copy holds `values.len()` values of [`DecimalDType::canonical_values_type`]. It never
/// rescales, so the values keep the scale of the source.
///
/// - A narrowing copy, where the canonical width is narrower than `F`, checks every valid value
///   against the precision of `target` with [`check_precision`]. A value that fits the width but
///   not the precision fails too. Null slots of the copy hold zeros.
/// - A widening or same-width copy checks nothing and keeps the null slots. The caller **must**
///   know that every valid value fits `target`, for example because `target` has at least the
///   precision of the source dtype. Otherwise the copy holds values outside the precision, which
///   is an incorrect result but not undefined behavior. A caller that owns a buffer that already
///   has the canonical width can keep it and call [`check_precision`] instead.
///
/// # Errors
///
/// Returns an error if `validity` and `values` have different lengths, or if a narrowing copy
/// finds a valid value outside the precision of `target`, as [`check_precision`] describes.
pub fn copy_to_canonical_width<F: NativeDecimalType>(
    values: &[F],
    validity: &Mask,
    target: DecimalDType,
) -> VortexResult<BufferHandle> {
    vortex_ensure_eq!(validity.len(), values.len());

    let canonical_type = target.canonical_values_type();
    if F::DECIMAL_TYPE == canonical_type {
        return Ok(BufferHandle::new_host(
            Buffer::<F>::copy_from(values).into_byte_buffer(),
        ));
    }

    let copy = match_each_decimal_value_type!(canonical_type, |W| {
        if const { (F::DECIMAL_TYPE as u8) <= (W::DECIMAL_TYPE as u8) } {
            widen_values::<F, W>(values).into_byte_buffer()
        } else {
            check_precision(values, validity, target)?;
            narrow_checked_values::<F, W>(values, validity).into_byte_buffer()
        }
    });

    Ok(BufferHandle::new_host(copy))
}

/// Widens every value to `W`, which **must** be at least as wide as `F`. Null slots keep their
/// values.
fn widen_values<F: NativeDecimalType, W: NativeDecimalType>(values: &[F]) -> Buffer<W> {
    values
        .iter()
        .map(|&value| {
            <W as BigCast>::from(value)
                .vortex_expect("W is at least as wide as F, so every value of F fits W")
        })
        .collect()
}

/// Narrows values that [`check_precision`] accepted to `W`, and writes zeros into the null slots.
fn narrow_checked_values<F: NativeDecimalType, W: NativeDecimalType>(
    values: &[F],
    validity: &Mask,
) -> Buffer<W> {
    let len = values.len();
    let valid_bits = match validity.bit_buffer() {
        AllOr::All => None,
        AllOr::None => return Buffer::zeroed(len),
        AllOr::Some(valid_bits) => Some(valid_bits),
    };

    // NB: a null slot can hold a value that does not fit `W`, so the map must not fail on it.
    // `map_masked_into` then overwrites every null slot with zero.
    let narrow = |value: F| <W as BigCast>::from(value).unwrap_or_default();

    let mut buffer = BufferMut::<W>::with_capacity(len);
    let out = &mut buffer.spare_capacity_mut()[..len];
    match valid_bits {
        None => values.map_into(out, narrow),
        Some(valid_bits) => values.map_masked_into(valid_bits, out, narrow),
    }

    // SAFETY: `map_into` and `map_masked_into` write every lane of `out`.
    unsafe { buffer.set_len(len) };

    buffer.freeze()
}

/// Checks that every valid value fits the precision of `decimal_dtype`.
///
/// Null slots are not checked, so they can hold any value. The check takes a materialized
/// [`Mask`], so it needs no execution context.
///
/// # Errors
///
/// Returns an error if `validity` and `values` have different lengths, or if a valid value is
/// outside the precision of `decimal_dtype`. The error names the first such row, its value and
/// `decimal_dtype`.
pub fn check_precision<T: NativeDecimalType>(
    values: &[T],
    validity: &Mask,
    decimal_dtype: DecimalDType,
) -> VortexResult<()> {
    vortex_ensure_eq!(validity.len(), values.len());

    // The bound tables stop at `T::MAX_PRECISION`. Every value of `T` has at most
    // `T::MAX_PRECISION + 1` digits, so a larger precision holds every value of `T`.
    let precision = decimal_dtype.precision();
    if precision > T::MAX_PRECISION {
        return Ok(());
    }
    let min = T::MIN_BY_PRECISION[usize::from(precision)];
    let max = T::MAX_BY_PRECISION[usize::from(precision)];
    let is_outside = |value: T| (value < min) | (value > max);

    let valid_bits = match validity.bit_buffer() {
        AllOr::All => None,
        AllOr::None => return Ok(()),
        AllOr::Some(valid_bits) => Some(valid_bits),
    };

    // The fold is branchless so that it vectorizes. It also reads the null slots, so a hit only
    // means that the rescan below must look for a valid row outside the precision.
    let any_outside = values
        .iter()
        .fold(false, |acc, &value| acc | is_outside(value));
    if !any_outside {
        return Ok(());
    }

    let first_outside = match valid_bits {
        None => values.iter().position(|&value| is_outside(value)),
        Some(valid_bits) => valid_bits
            .set_indices()
            .find(|&row| is_outside(values[row])),
    };
    if let Some(row) = first_outside {
        vortex_bail!(
            "decimal value at row {row} must be within the precision of {decimal_dtype}, got {}",
            values[row]
        );
    }

    Ok(())
}

macro_rules! try_downcast {
    ($array:expr, from: $src:ty, to: $($dst:ty),*) => {{
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
    use rstest::rstest;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;

    use super::check_precision;
    use super::converted_buffer;
    use super::copy_to_canonical_width;
    use crate::arrays::DecimalArray;
    use crate::dtype::BigCast;
    use crate::dtype::DecimalDType;
    use crate::dtype::DecimalType;
    use crate::dtype::NativeDecimalType;
    use crate::dtype::ToI256;
    use crate::dtype::i256;
    use crate::match_each_decimal_value_type;
    use crate::validity::Validity;

    /// Runs [`copy_to_canonical_width`], checks that the copy has the canonical width of `target`,
    /// and reads the copy back as `i256`.
    fn copy_as_i256<F: NativeDecimalType>(
        values: &[F],
        validity: &Mask,
        target: DecimalDType,
    ) -> VortexResult<Vec<i256>> {
        let copy = copy_to_canonical_width(values, validity, target)?;
        let width = target.canonical_values_type();
        assert_eq!(copy.len(), values.len() * width.byte_width());

        Ok(match_each_decimal_value_type!(width, |W| {
            Buffer::<W>::from_byte_buffer(copy.unwrap_host())
                .iter()
                .map(|value| value.to_i256().vortex_expect("decimal values fit i256"))
                .collect()
        }))
    }

    /// Returns `10^precision`, the smallest magnitude outside `precision`.
    fn ten_to_the(precision: u8) -> i256 {
        i256::from_i128(10)
            .checked_pow(precision.into())
            .vortex_expect("10^76 fits i256")
    }

    /// Converts each test value to the native type `T`.
    fn to_native<T: NativeDecimalType>(values: &[i256]) -> Vec<T> {
        values
            .iter()
            .map(|&value| <T as BigCast>::from(value).vortex_expect("test value fits the type"))
            .collect()
    }

    /// Each pair selects a different widening or narrowing conversion.
    #[rstest]
    fn converted_buffer_every_storage_pair(
        #[values(
            DecimalType::I8,
            DecimalType::I16,
            DecimalType::I32,
            DecimalType::I64,
            DecimalType::I128,
            DecimalType::I256
        )]
        storage: DecimalType,
        #[values(
            DecimalType::I8,
            DecimalType::I16,
            DecimalType::I32,
            DecimalType::I64,
            DecimalType::I128,
            DecimalType::I256
        )]
        target: DecimalType,
    ) -> VortexResult<()> {
        let values = [1i8, -2, 3];
        let dtype = DecimalDType::new(2, 0);
        let array = match_each_decimal_value_type!(storage, |S| {
            let values = values.map(|value| {
                <S as BigCast>::from(value).vortex_expect("small values fit every decimal type")
            });
            DecimalArray::new(Buffer::copy_from(values), dtype, Validity::NonNullable)
        });

        match_each_decimal_value_type!(target, |W| {
            let converted = converted_buffer::<W>(&array, &Mask::new_true(values.len()))?;
            let expected = values.map(|value| {
                <W as BigCast>::from(value).vortex_expect("small values fit every decimal type")
            });
            assert_eq!(converted.as_slice(), expected.as_slice());
        });
        Ok(())
    }

    #[rstest]
    #[case::i8_to_i16(DecimalType::I8, 3)]
    #[case::i16_to_i32(DecimalType::I16, 5)]
    #[case::i32_to_i64(DecimalType::I32, 10)]
    #[case::i64_to_i128(DecimalType::I64, 19)]
    #[case::i128_to_i256(DecimalType::I128, 39)]
    #[case::i8_to_i256(DecimalType::I8, 76)]
    #[case::i64_same_width(DecimalType::I64, 18)]
    fn test_copy_to_canonical_width_widens(
        #[case] source: DecimalType,
        #[case] target_precision: u8,
    ) -> VortexResult<()> {
        let target = DecimalDType::new(target_precision, 2);
        match_each_decimal_value_type!(source, |F| {
            // The extremes of the largest precision that `F` holds, and zero.
            let limit = ten_to_the(F::MAX_PRECISION) - i256::ONE;
            let expected = [-limit, i256::ZERO, limit];
            let values = to_native::<F>(&expected);
            assert_eq!(copy_as_i256(&values, &Mask::new_true(3), target)?, expected);
        });
        Ok(())
    }

    /// Each case puts `10^p` inside the target width, so only the precision check can reject it.
    #[rstest]
    #[case::i16_to_i8(DecimalType::I16, 2)]
    #[case::i32_to_i16(DecimalType::I32, 4)]
    #[case::i64_to_i32(DecimalType::I64, 9)]
    #[case::i128_to_i64(DecimalType::I128, 18)]
    #[case::i256_to_i128(DecimalType::I256, 38)]
    #[case::i256_to_i8(DecimalType::I256, 1)]
    fn test_copy_to_canonical_width_narrowing_boundaries(
        #[case] source: DecimalType,
        #[case] target_precision: u8,
    ) -> VortexResult<()> {
        let target = DecimalDType::new(target_precision, 0);
        let limit = ten_to_the(target_precision);
        let inside = [-(limit - i256::ONE), limit - i256::ONE];
        let outside = [-limit, limit];

        match_each_decimal_value_type!(source, |F| {
            let values = to_native::<F>(&inside);
            assert_eq!(copy_as_i256(&values, &Mask::new_true(2), target)?, inside);

            for value in to_native::<F>(&outside) {
                assert!(copy_to_canonical_width(&[value], &Mask::new_true(1), target).is_err());
            }
        });
        Ok(())
    }

    #[test]
    fn test_copy_to_canonical_width_zeros_null_slots() -> VortexResult<()> {
        // Row 1 holds a null value outside the `i32` width, and row 3 a null value inside the
        // width but outside the precision.
        let values = [12_345i128, 10i128.pow(30), -99_999, 100_000];
        let validity = Mask::from_iter([true, false, true, false]);

        let copy = copy_as_i256(&values, &validity, DecimalDType::new(5, 2))?;
        assert_eq!(copy, [12_345, 0, -99_999, 0].map(i256::from_i128));
        Ok(())
    }

    #[test]
    fn test_copy_to_canonical_width_error_names_row() {
        // Row 1 is a null slot outside the precision, so row 2 is the first failing valid row.
        let values = [1i64, 1_000_000, 1_000, 5_000];
        let validity = Mask::from_iter([true, false, true, true]);

        let error = copy_to_canonical_width(&values, &validity, DecimalDType::new(3, 0))
            .expect_err("row 2 is outside the precision")
            .to_string();
        assert!(error.contains("row 2"), "{error}");
        assert!(error.contains("got 1000"), "{error}");
        assert!(error.contains("decimal(3,0)"), "{error}");
    }

    #[test]
    fn test_copy_to_canonical_width_all_null() -> VortexResult<()> {
        let values = [10i128.pow(30), i128::MIN, 7];
        let copy = copy_as_i256(&values, &Mask::new_false(3), DecimalDType::new(5, 2))?;
        assert_eq!(copy, [i256::ZERO; 3]);
        Ok(())
    }

    /// Precision 4 narrows the `i32` values, and precision 18 widens them.
    #[rstest]
    fn test_copy_to_canonical_width_length_mismatch(#[values(4, 18)] target_precision: u8) {
        let target = DecimalDType::new(target_precision, 0);
        let result = copy_to_canonical_width(&[1i32, 2, 3], &Mask::new_true(2), target);
        assert!(result.is_err());
    }

    /// Each case puts `10^p` inside the width of the values, so only the precision rejects it.
    #[rstest]
    #[case::i8(DecimalType::I8, 2)]
    #[case::i16(DecimalType::I16, 4)]
    #[case::i32(DecimalType::I32, 9)]
    #[case::i64(DecimalType::I64, 18)]
    #[case::i128(DecimalType::I128, 38)]
    #[case::i256(DecimalType::I256, 76)]
    #[case::i128_below_max_precision(DecimalType::I128, 5)]
    fn test_check_precision_boundaries(
        #[case] values_type: DecimalType,
        #[case] precision: u8,
    ) -> VortexResult<()> {
        let decimal_dtype = DecimalDType::new(precision, 0);
        let limit = ten_to_the(precision);
        let inside = [-(limit - i256::ONE), limit - i256::ONE];
        let outside = [-limit, limit];

        match_each_decimal_value_type!(values_type, |T| {
            check_precision(&to_native::<T>(&inside), &Mask::new_true(2), decimal_dtype)?;

            for value in to_native::<T>(&outside) {
                assert!(check_precision(&[value], &Mask::new_true(1), decimal_dtype).is_err());
            }
        });
        Ok(())
    }

    /// Every value of a type has at most one digit more than its `MAX_PRECISION`.
    #[test]
    fn test_check_precision_above_max_precision_of_values_type() -> VortexResult<()> {
        let all_valid = Mask::new_true(2);
        check_precision(&[i8::MIN, i8::MAX], &all_valid, DecimalDType::new(3, 0))?;
        check_precision(&[i64::MIN, i64::MAX], &all_valid, DecimalDType::new(19, 0))?;
        Ok(())
    }

    #[test]
    fn test_check_precision_ignores_null_slots() -> VortexResult<()> {
        let values = [5i32, 1_000_000_000, -7];
        let validity = Mask::from_iter([true, false, true]);
        let decimal_dtype = DecimalDType::new(4, 0);

        check_precision(&values, &validity, decimal_dtype)?;
        check_precision(&values, &Mask::new_false(3), decimal_dtype)?;
        Ok(())
    }
}
