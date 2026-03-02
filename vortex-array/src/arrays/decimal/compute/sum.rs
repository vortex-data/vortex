// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use num_traits::AsPrimitive;
use num_traits::CheckedAdd;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;

use crate::arrays::DecimalArray;
use crate::arrays::DecimalVTable;
use crate::compute::SumKernel;
use crate::compute::SumKernelAdapter;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::Nullability::Nullable;
use crate::expr::stats::Stat;
use crate::match_each_decimal_value_type;
use crate::register_kernel;
use crate::scalar::DecimalValue;
use crate::scalar::Scalar;

impl SumKernel for DecimalVTable {
    fn sum(&self, array: &DecimalArray, accumulator: &Scalar) -> VortexResult<Scalar> {
        let return_dtype = Stat::Sum
            .dtype(array.dtype())
            .vortex_expect("sum for decimals exists");
        let return_decimal_dtype = *return_dtype
            .as_decimal_opt()
            .vortex_expect("must be decimal");

        // Extract the initial value as a `DecimalValue`.
        let initial_decimal = accumulator
            .as_decimal()
            .decimal_value()
            .vortex_expect("cannot be null");

        let mask = array.validity_mask()?;
        let validity = match &mask {
            Mask::AllTrue(_) => None,
            Mask::Values(mask_values) => Some(mask_values.bit_buffer()),
            Mask::AllFalse(_) => {
                vortex_bail!("invalid state, all-null array should be checked by top-level sum fn")
            }
        };

        let values_type = DecimalType::smallest_decimal_value_type(&return_decimal_dtype);
        match_each_decimal_value_type!(array.values_type(), |I| {
            match_each_decimal_value_type!(values_type, |O| {
                let initial_val: O = initial_decimal
                    .cast()
                    .vortex_expect("cannot fail to cast initial value");

                Ok(sum_to_scalar(
                    array.buffer::<I>(),
                    validity,
                    initial_val,
                    return_decimal_dtype,
                    &return_dtype,
                ))
            })
        })
    }
}

/// Compute the checked sum and convert the result to a [`Scalar`].
///
/// Returns a null scalar if the sum overflows the underlying integer type or if the result
/// exceeds the declared decimal precision.
fn sum_to_scalar<T, O>(
    values: Buffer<T>,
    validity: Option<&BitBuffer>,
    initial: O,
    return_decimal_dtype: DecimalDType,
    return_dtype: &DType,
) -> Scalar
where
    T: AsPrimitive<O>,
    O: Copy + CheckedAdd + Into<DecimalValue> + 'static,
{
    let raw_sum = match validity {
        Some(v) => sum_decimal_with_validity(values, v, initial),
        None => sum_decimal(values, initial),
    };

    raw_sum
        .map(Into::<DecimalValue>::into)
        // We have to make sure that the decimal value fits the precision of the decimal dtype.
        .filter(|v| v.fits_in_precision(return_decimal_dtype))
        .map(|v| Scalar::decimal(v, return_decimal_dtype, Nullable))
        // If an overflow occurs during summation, or final value does not fit, then return a null.
        .unwrap_or_else(|| Scalar::null(return_dtype.clone()))
}

fn sum_decimal<T: AsPrimitive<I>, I: Copy + CheckedAdd + 'static>(
    values: Buffer<T>,
    initial: I,
) -> Option<I> {
    let mut sum = initial;
    for v in values.iter() {
        let v: I = v.as_();
        sum = CheckedAdd::checked_add(&sum, &v)?;
    }
    Some(sum)
}

fn sum_decimal_with_validity<T: AsPrimitive<I>, I: Copy + CheckedAdd + 'static>(
    values: Buffer<T>,
    validity: &BitBuffer,
    initial: I,
) -> Option<I> {
    let mut sum = initial;
    for (v, valid) in values.iter().zip_eq(validity) {
        if valid {
            let v: I = v.as_();
            sum = CheckedAdd::checked_add(&sum, &v)?;
        }
    }
    Some(sum)
}

register_kernel!(SumKernelAdapter(DecimalVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;

    use crate::arrays::DecimalArray;
    use crate::compute::sum;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::i256;
    use crate::scalar::DecimalValue;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::validity::Validity;

    #[test]
    fn test_sum_basic() {
        let decimal = DecimalArray::new(
            buffer![100i32, 200i32, 300i32],
            DecimalDType::new(4, 2),
            Validity::AllValid,
        );

        let result = sum(&decimal.to_array()).unwrap();

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(14, 2), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(600i32))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sum_with_nulls() {
        let decimal = DecimalArray::new(
            buffer![100i32, 200i32, 300i32, 400i32],
            DecimalDType::new(4, 2),
            Validity::from_iter([true, false, true, true]),
        );

        let result = sum(&decimal.to_array()).unwrap();

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(14, 2), Nullability::Nullable),
            Some(ScalarValue::from(DecimalValue::from(800i32))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sum_negative_values() {
        let decimal = DecimalArray::new(
            buffer![100i32, -200i32, 300i32, -50i32],
            DecimalDType::new(4, 2),
            Validity::AllValid,
        );

        let result = sum(&decimal.to_array()).unwrap();

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(14, 2), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(150i32))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sum_near_i32_max() {
        // Test values close to i32::MAX to ensure proper handling
        let near_max = i32::MAX - 1000;
        let decimal = DecimalArray::new(
            buffer![near_max, 500i32, 400i32],
            DecimalDType::new(10, 2),
            Validity::AllValid,
        );

        let result = sum(&decimal.to_array()).unwrap();

        // Should use i64 for accumulation since precision increases
        let expected_sum = near_max as i64 + 500 + 400;
        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(20, 2), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(expected_sum))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sum_large_i64_values() {
        // Test with large i64 values that require i128 accumulation
        let large_val = i64::MAX / 4;
        let decimal = DecimalArray::new(
            buffer![large_val, large_val, large_val, large_val + 1],
            DecimalDType::new(19, 0),
            Validity::AllValid,
        );

        let result = sum(&decimal.to_array()).unwrap();

        let expected_sum = (large_val as i128) * 4 + 1;
        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(29, 0), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(expected_sum))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sum_overflow_detection() {
        use crate::dtype::i256;

        // Create values that will overflow when summed
        // Use maximum i128 values that will overflow when added
        let max_val = i128::MAX / 2;
        let decimal = DecimalArray::new(
            buffer![max_val, max_val, max_val],
            DecimalDType::new(38, 0),
            Validity::AllValid,
        );

        let result = sum(&decimal.to_array()).unwrap();

        // Should use i256 for accumulation
        let expected_sum =
            i256::from_i128(max_val) + i256::from_i128(max_val) + i256::from_i128(max_val);
        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(48, 0), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(expected_sum))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sum_mixed_signs_near_overflow() {
        // Test that mixed signs work correctly near overflow boundaries
        let large_pos = i64::MAX / 2;
        let large_neg = -(i64::MAX / 2);
        let decimal = DecimalArray::new(
            buffer![large_pos, large_neg, large_pos, 1000i64],
            DecimalDType::new(19, 3),
            Validity::AllValid,
        );

        let result = sum(&decimal.to_array()).unwrap();

        let expected_sum = (large_pos as i128) + (large_neg as i128) + (large_pos as i128) + 1000;
        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(29, 3), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(expected_sum))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sum_preserves_scale() {
        let decimal = DecimalArray::new(
            buffer![12345i32, 67890i32, 11111i32],
            DecimalDType::new(6, 4),
            Validity::AllValid,
        );

        let result = sum(&decimal.to_array()).unwrap();

        // Scale should be preserved, precision increased by 10
        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(16, 4), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(91346i32))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sum_single_value() {
        let decimal =
            DecimalArray::new(buffer![42i32], DecimalDType::new(3, 1), Validity::AllValid);

        let result = sum(&decimal.to_array()).unwrap();

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(13, 1), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(42i32))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sum_with_all_nulls_except_one() {
        let decimal = DecimalArray::new(
            buffer![100i32, 200i32, 300i32, 400i32],
            DecimalDType::new(4, 2),
            Validity::from_iter([false, false, true, false]),
        );

        let result = sum(&decimal.to_array()).unwrap();

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(14, 2), Nullability::Nullable),
            Some(ScalarValue::from(DecimalValue::from(300i32))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sum_i128_to_i256_boundary() {
        // Test the boundary between i128 and i256 accumulation
        let large_i128 = i128::MAX / 10;
        let decimal = DecimalArray::new(
            buffer![
                large_i128, large_i128, large_i128, large_i128, large_i128, large_i128, large_i128,
                large_i128, large_i128
            ],
            DecimalDType::new(38, 0),
            Validity::AllValid,
        );

        let result = sum(&decimal.to_array()).unwrap();

        // Should use i256 for accumulation since 9 * (i128::MAX / 10) fits in i128 but we increase precision
        let expected_sum = i256::from_i128(large_i128).wrapping_pow(1) * i256::from_i128(9);
        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(48, 0), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(expected_sum))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sum_precision_overflow_without_i256_overflow() {
        // Construct values that individually fit in precision 76 but whose sum exceeds it,
        // while still fitting in `i256`. This ensures we return null for precision overflow
        // and not just for arithmetic overflow.
        let ten_to_38 = i256::from_i128(10i128.pow(38));
        let ten_to_75 = ten_to_38 * i256::from_i128(10i128.pow(37));
        // 6 * 10^75 is a 76-digit number, which fits in precision 76.
        let val = ten_to_75 * i256::from_i128(6);

        let decimal_dtype = DecimalDType::new(76, 0);
        let decimal = DecimalArray::new(buffer![val, val], decimal_dtype, Validity::AllValid);

        // Sum = 12 * 10^75 = 1.2 * 10^76, which exceeds precision 76 but fits in `i256`.
        let result = sum(&decimal.to_array()).unwrap();
        assert_eq!(
            result,
            Scalar::null(DType::Decimal(decimal_dtype, Nullability::Nullable))
        );
    }

    #[test]
    fn test_i256_overflow() {
        let decimal_dtype = DecimalDType::new(76, 0);
        let decimal = DecimalArray::new(
            buffer![i256::MAX, i256::MAX, i256::MAX],
            decimal_dtype,
            Validity::AllValid,
        );

        assert_eq!(
            sum(&decimal.to_array()).vortex_expect("operation should succeed in test"),
            Scalar::null(DType::Decimal(decimal_dtype, Nullability::Nullable))
        );
    }
}
