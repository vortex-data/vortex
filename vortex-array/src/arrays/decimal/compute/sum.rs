// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use arrow_schema::DECIMAL256_MAX_PRECISION;
use num_traits::AsPrimitive;
use vortex_dtype::Nullability::Nullable;
use vortex_dtype::{DecimalDType, DecimalType};
use vortex_error::{VortexResult, vortex_bail};
use vortex_mask::Mask;
use vortex_scalar::{DecimalValue, Scalar, match_each_decimal_value_type};

use crate::arrays::{DecimalArray, DecimalVTable};
use crate::compute::{SumKernel, SumKernelAdapter};
use crate::register_kernel;

// Its safe to use `AsPrimitive` here because we always cast up.
macro_rules! sum_decimal {
    ($ty:ty, $values:expr) => {{
        let mut sum: $ty = <$ty>::default();
        for v in $values.iter() {
            let v: $ty = (*v).as_();
            sum += v;
        }
        sum
    }};
    ($ty:ty, $values:expr, $validity:expr) => {{
        use itertools::Itertools;

        let mut sum: $ty = <$ty>::default();
        for (v, valid) in $values.iter().zip_eq($validity) {
            if valid {
                let v: $ty = (*v).as_();
                sum += v;
            }
        }
        sum
    }};
}

impl SumKernel for DecimalVTable {
    #[allow(clippy::cognitive_complexity)]
    fn sum(&self, array: &DecimalArray) -> VortexResult<Scalar> {
        let decimal_dtype = array.decimal_dtype();

        // Both Spark and DataFusion use this heuristic.
        // - https://github.com/apache/spark/blob/fcf636d9eb8d645c24be3db2d599aba2d7e2955a/sql/catalyst/src/main/scala/org/apache/spark/sql/catalyst/expressions/aggregate/Sum.scala#L66
        // - https://github.com/apache/datafusion/blob/4153adf2c0f6e317ef476febfdc834208bd46622/datafusion/functions-aggregate/src/sum.rs#L188
        let new_precision = u8::min(DECIMAL256_MAX_PRECISION, decimal_dtype.precision() + 10);
        let new_scale = decimal_dtype.scale();
        let return_dtype = DecimalDType::new(new_precision, new_scale);

        match array.validity_mask() {
            Mask::AllFalse(_) => {
                vortex_bail!("invalid state, all-null array should be checked by top-level sum fn")
            }
            Mask::AllTrue(_) => {
                let values_type = DecimalType::smallest_decimal_value_type(&return_dtype);
                match_each_decimal_value_type!(array.values_type(), |I| {
                    match_each_decimal_value_type!(values_type, |O| {
                        Ok(Scalar::decimal(
                            DecimalValue::from(sum_decimal!(O, array.buffer::<I>())),
                            return_dtype,
                            Nullable,
                        ))
                    })
                })
            }
            Mask::Values(mask_values) => {
                let values_type = DecimalType::smallest_decimal_value_type(&return_dtype);
                match_each_decimal_value_type!(array.values_type(), |I| {
                    match_each_decimal_value_type!(values_type, |O| {
                        Ok(Scalar::decimal(
                            DecimalValue::from(sum_decimal!(
                                O,
                                array.buffer::<I>(),
                                mask_values.bit_buffer()
                            )),
                            return_dtype,
                            Nullable,
                        ))
                    })
                })
            }
        }
    }
}

register_kernel!(SumKernelAdapter(DecimalVTable).lift());

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;
    use vortex_dtype::{DType, DecimalDType, Nullability};
    use vortex_scalar::{DecimalValue, Scalar, ScalarValue};

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
        use vortex_scalar::i256;

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
}
