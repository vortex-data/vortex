// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use itertools::Itertools;
use num_traits::AsPrimitive;
use num_traits::CheckedAdd;
use vortex_buffer::BitBuffer;
use vortex_buffer::Buffer;
use vortex_dtype::DecimalType;
use vortex_dtype::Nullability::Nullable;
use vortex_dtype::match_each_decimal_value_type;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_scalar::DecimalScalar;
use vortex_scalar::DecimalValue;
use vortex_scalar::Scalar;

use crate::arrays::DecimalArray;
use crate::arrays::DecimalVTable;
use crate::compute::SumKernel;
use crate::compute::SumKernelAdapter;
use crate::register_kernel;
use crate::stats::Stat;

impl SumKernel for DecimalVTable {
    #[expect(
        clippy::cognitive_complexity,
        reason = "complexity from nested match_each_* macros"
    )]
    fn sum(&self, array: &DecimalArray, accumulator: &Scalar) -> VortexResult<Scalar> {
        let return_dtype = Stat::Sum
            .dtype(array.dtype())
            .vortex_expect("sum for decimals exists");
        let return_decimal_dtype = return_dtype
            .as_decimal_opt()
            .vortex_expect("must be decimal");

        // Extract the initial value as a DecimalValue
        let initial_decimal = DecimalScalar::try_from(accumulator)
            .vortex_expect("must be a decimal")
            .decimal_value()
            .vortex_expect("cannot be null");

        match array.validity_mask() {
            Mask::AllFalse(_) => {
                vortex_bail!("invalid state, all-null array should be checked by top-level sum fn")
            }
            Mask::AllTrue(_) => {
                let values_type = DecimalType::smallest_decimal_value_type(return_decimal_dtype);
                match_each_decimal_value_type!(array.values_type(), |I| {
                    match_each_decimal_value_type!(values_type, |O| {
                        let initial_val: O = initial_decimal
                            .cast()
                            .vortex_expect("cannot fail to cast initial value");
                        if let Some(sum) = sum_decimal(array.buffer::<I>(), initial_val) {
                            Ok(Scalar::decimal(
                                DecimalValue::from(sum),
                                *return_decimal_dtype,
                                Nullable,
                            ))
                        } else {
                            Ok(Scalar::null(return_dtype))
                        }
                    })
                })
            }
            Mask::Values(mask_values) => {
                let values_type = DecimalType::smallest_decimal_value_type(return_decimal_dtype);
                match_each_decimal_value_type!(array.values_type(), |I| {
                    match_each_decimal_value_type!(values_type, |O| {
                        let initial_val: O = initial_decimal
                            .cast()
                            .vortex_expect("cannot fail to cast initial value");

                        if let Some(sum) = sum_decimal_with_validity(
                            array.buffer::<I>(),
                            mask_values.bit_buffer(),
                            initial_val,
                        ) {
                            Ok(Scalar::decimal(
                                DecimalValue::from(sum),
                                *return_decimal_dtype,
                                Nullable,
                            ))
                        } else {
                            Ok(Scalar::null(return_dtype))
                        }
                    })
                })
            }
        }
    }
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
    use vortex_dtype::DType;
    use vortex_dtype::DecimalDType;
    use vortex_dtype::Nullability;
    use vortex_error::VortexUnwrap;
    use vortex_scalar::DecimalValue;
    use vortex_scalar::Scalar;
    use vortex_scalar::ScalarValue;
    use vortex_scalar::i256;

    use crate::arrays::DecimalArray;
    use crate::compute::sum;
    use crate::validity::Validity;

    #[test]
    fn test_sum_basic() {
        let decimal = DecimalArray::new(
            buffer![100i32, 200i32, 300i32],
            DecimalDType::new(4, 2),
            Validity::AllValid,
        );

        let result = sum(decimal.as_ref()).unwrap();

        let expected = Scalar::new(
            DType::Decimal(DecimalDType::new(14, 2), Nullability::NonNullable),
            ScalarValue::from(DecimalValue::from(600i32)),
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sum_with_nulls() {
        let decimal = DecimalArray::new(
            buffer![100i32, 200i32, 300i32, 400i32],
            DecimalDType::new(4, 2),
            Validity::from_iter([true, false, true, true]),
        );

        let result = sum(decimal.as_ref()).unwrap();

        let expected = Scalar::new(
            DType::Decimal(DecimalDType::new(14, 2), Nullability::Nullable),
            ScalarValue::from(DecimalValue::from(800i32)),
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sum_negative_values() {
        let decimal = DecimalArray::new(
            buffer![100i32, -200i32, 300i32, -50i32],
            DecimalDType::new(4, 2),
            Validity::AllValid,
        );

        let result = sum(decimal.as_ref()).unwrap();

        let expected = Scalar::new(
            DType::Decimal(DecimalDType::new(14, 2), Nullability::NonNullable),
            ScalarValue::from(DecimalValue::from(150i32)),
        );

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

        let result = sum(decimal.as_ref()).unwrap();

        // Should use i64 for accumulation since precision increases
        let expected_sum = near_max as i64 + 500 + 400;
        let expected = Scalar::new(
            DType::Decimal(DecimalDType::new(20, 2), Nullability::NonNullable),
            ScalarValue::from(DecimalValue::from(expected_sum)),
        );

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

        let result = sum(decimal.as_ref()).unwrap();

        let expected_sum = (large_val as i128) * 4 + 1;
        let expected = Scalar::new(
            DType::Decimal(DecimalDType::new(29, 0), Nullability::NonNullable),
            ScalarValue::from(DecimalValue::from(expected_sum)),
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sum_overflow_detection() {
        use vortex_scalar::i256;

        // Create values that will overflow when summed
        // Use maximum i128 values that will overflow when added
        let max_val = i128::MAX / 2;
        let decimal = DecimalArray::new(
            buffer![max_val, max_val, max_val],
            DecimalDType::new(38, 0),
            Validity::AllValid,
        );

        let result = sum(decimal.as_ref()).unwrap();

        // Should use i256 for accumulation
        let expected_sum =
            i256::from_i128(max_val) + i256::from_i128(max_val) + i256::from_i128(max_val);
        let expected = Scalar::new(
            DType::Decimal(DecimalDType::new(48, 0), Nullability::NonNullable),
            ScalarValue::from(DecimalValue::from(expected_sum)),
        );

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

        let result = sum(decimal.as_ref()).unwrap();

        let expected_sum = (large_pos as i128) + (large_neg as i128) + (large_pos as i128) + 1000;
        let expected = Scalar::new(
            DType::Decimal(DecimalDType::new(29, 3), Nullability::NonNullable),
            ScalarValue::from(DecimalValue::from(expected_sum)),
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sum_preserves_scale() {
        let decimal = DecimalArray::new(
            buffer![12345i32, 67890i32, 11111i32],
            DecimalDType::new(6, 4),
            Validity::AllValid,
        );

        let result = sum(decimal.as_ref()).unwrap();

        // Scale should be preserved, precision increased by 10
        let expected = Scalar::new(
            DType::Decimal(DecimalDType::new(16, 4), Nullability::NonNullable),
            ScalarValue::from(DecimalValue::from(91346i32)),
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sum_single_value() {
        let decimal =
            DecimalArray::new(buffer![42i32], DecimalDType::new(3, 1), Validity::AllValid);

        let result = sum(decimal.as_ref()).unwrap();

        let expected = Scalar::new(
            DType::Decimal(DecimalDType::new(13, 1), Nullability::NonNullable),
            ScalarValue::from(DecimalValue::from(42i32)),
        );

        assert_eq!(result, expected);
    }

    #[test]
    fn test_sum_with_all_nulls_except_one() {
        let decimal = DecimalArray::new(
            buffer![100i32, 200i32, 300i32, 400i32],
            DecimalDType::new(4, 2),
            Validity::from_iter([false, false, true, false]),
        );

        let result = sum(decimal.as_ref()).unwrap();

        let expected = Scalar::new(
            DType::Decimal(DecimalDType::new(14, 2), Nullability::Nullable),
            ScalarValue::from(DecimalValue::from(300i32)),
        );

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

        let result = sum(decimal.as_ref()).unwrap();

        // Should use i256 for accumulation since 9 * (i128::MAX / 10) fits in i128 but we increase precision
        let expected_sum = i256::from_i128(large_i128).wrapping_pow(1) * i256::from_i128(9);
        let expected = Scalar::new(
            DType::Decimal(DecimalDType::new(48, 0), Nullability::NonNullable),
            ScalarValue::from(DecimalValue::from(expected_sum)),
        );

        assert_eq!(result, expected);
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
            sum(decimal.as_ref()).vortex_unwrap(),
            Scalar::null(DType::Decimal(decimal_dtype, Nullability::Nullable))
        );
    }
}
