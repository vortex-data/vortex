// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::ArrayRef;
use crate::aggregate_fn::Accumulator;
use crate::aggregate_fn::DynAccumulator;
use crate::aggregate_fn::EmptyOptions;
use crate::aggregate_fn::fns::sum::Sum;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::scalar::Scalar;

/// Sum an array, starting from zero.
///
/// If the sum overflows, a null scalar will be returned.
/// If the sum is not supported for the array's dtype, an error will be raised.
/// If the array is all-invalid, the sum will be zero.
pub fn sum(array: &ArrayRef) -> VortexResult<Scalar> {
    // Validate that sum is supported for this dtype.
    Stat::Sum
        .dtype(array.dtype())
        .ok_or_else(|| vortex_err!("Sum not supported for dtype: {}", array.dtype()))?;

    // Short-circuit using cached array statistics.
    if let Some(Precision::Exact(sum_scalar)) = array.statistics().get(Stat::Sum) {
        return Ok(sum_scalar);
    }

    // Compute using Accumulator<Sum>.
    let mut acc = Accumulator::try_new(
        Sum,
        EmptyOptions,
        array.dtype().clone(),
        VortexSession::empty(),
    )?;
    acc.accumulate(array)?;
    let result = acc.finish()?;

    // Cache the computed sum as a statistic (only if non-null, i.e. no overflow).
    if let Some(val) = result.value().cloned() {
        array.statistics().set(Stat::Sum, Precision::Exact(val));
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use num_traits::CheckedAdd;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;

    use crate::ArrayRef;
    use crate::DynArray;
    use crate::IntoArray as _;
    use crate::arrays::BoolArray;
    use crate::arrays::ChunkedArray;
    use crate::arrays::ConstantArray;
    use crate::arrays::DecimalArray;
    use crate::arrays::PrimitiveArray;
    use crate::compute::sum;
    use crate::dtype::DType;
    use crate::dtype::DecimalDType;
    use crate::dtype::Nullability;
    use crate::dtype::Nullability::Nullable;
    use crate::dtype::PType;
    use crate::dtype::i256;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::expr::stats::StatsProvider;
    use crate::scalar::DecimalValue;
    use crate::scalar::NumericOperator;
    use crate::scalar::Scalar;
    use crate::scalar::ScalarValue;
    use crate::validity::Validity;

    /// Sum an array with an initial value (test-only helper).
    fn sum_with_accumulator(array: &ArrayRef, accumulator: &Scalar) -> VortexResult<Scalar> {
        if accumulator.is_null() {
            return Ok(accumulator.clone());
        }
        if accumulator.is_zero() == Some(true) {
            return sum(array);
        }

        let sum_dtype = Stat::Sum.dtype(array.dtype()).ok_or_else(|| {
            vortex_error::vortex_err!("Sum not supported for dtype: {}", array.dtype())
        })?;

        // For non-float types, try statistics short-circuit with accumulator.
        if !matches!(&sum_dtype, DType::Primitive(p, _) if p.is_float())
            && let Some(Precision::Exact(sum_scalar)) = array.statistics().get(Stat::Sum)
        {
            return add_scalars(&sum_dtype, &sum_scalar, accumulator);
        }

        // Compute array sum from zero (also caches stats).
        let array_sum = sum(array)?;

        // Combine with the accumulator.
        add_scalars(&sum_dtype, &array_sum, accumulator)
    }

    /// Add two sum scalars with overflow checking.
    fn add_scalars(sum_dtype: &DType, lhs: &Scalar, rhs: &Scalar) -> VortexResult<Scalar> {
        if lhs.is_null() || rhs.is_null() {
            return Ok(Scalar::null(sum_dtype.as_nullable()));
        }

        Ok(match sum_dtype {
            DType::Primitive(ptype, _) if ptype.is_float() => {
                let lhs_val = f64::try_from(lhs)?;
                let rhs_val = f64::try_from(rhs)?;
                Scalar::primitive(lhs_val + rhs_val, Nullable)
            }
            DType::Primitive(..) => lhs
                .as_primitive()
                .checked_add(&rhs.as_primitive())
                .map(Scalar::from)
                .unwrap_or_else(|| Scalar::null(sum_dtype.as_nullable())),
            DType::Decimal(..) => lhs
                .as_decimal()
                .checked_binary_numeric(&rhs.as_decimal(), NumericOperator::Add)
                .map(Scalar::from)
                .unwrap_or_else(|| Scalar::null(sum_dtype.as_nullable())),
            _ => unreachable!("Sum will always be a decimal or a primitive dtype"),
        })
    }

    #[test]
    fn sum_all_invalid() {
        let array = PrimitiveArray::from_option_iter::<i32, _>([None, None, None]).into_array();
        let result = sum(&array).unwrap();
        assert_eq!(result, Scalar::primitive(0i64, Nullable));
    }

    #[test]
    fn sum_all_invalid_float() {
        let array = PrimitiveArray::from_option_iter::<f32, _>([None, None, None]).into_array();
        let result = sum(&array).unwrap();
        assert_eq!(result, Scalar::primitive(0f64, Nullable));
    }

    #[test]
    fn sum_constant() {
        let array = buffer![1, 1, 1, 1].into_array();
        let result = sum(&array).unwrap();
        assert_eq!(result.as_primitive().as_::<i32>(), Some(4));
    }

    #[test]
    fn sum_constant_float() {
        let array = buffer![1., 1., 1., 1.].into_array();
        let result = sum(&array).unwrap();
        assert_eq!(result.as_primitive().as_::<f32>(), Some(4.));
    }

    #[test]
    fn sum_boolean() {
        let array = BoolArray::from_iter([true, false, false, true]).into_array();
        let result = sum(&array).unwrap();
        assert_eq!(result.as_primitive().as_::<i32>(), Some(2));
    }

    #[test]
    fn sum_stats() {
        let array = ChunkedArray::try_new(
            vec![
                PrimitiveArray::from_iter([1, 1, 1]).into_array(),
                PrimitiveArray::from_iter([2, 2, 2]).into_array(),
            ],
            DType::Primitive(PType::I32, Nullability::NonNullable),
        )
        .vortex_expect("operation should succeed in test");
        let array = array.into_array();
        // compute sum with accumulator to populate stats
        sum_with_accumulator(&array, &Scalar::primitive(2i64, Nullable)).unwrap();

        let sum_without_acc = sum(&array).unwrap();
        assert_eq!(sum_without_acc, Scalar::primitive(9i64, Nullable));
    }

    // -- Constant array tests (migrated from constant/compute/sum.rs) --

    #[test]
    fn sum_constant_unsigned() {
        let array = ConstantArray::new(5u64, 10).into_array();
        let result = sum(&array).unwrap();
        assert_eq!(result, 50u64.into());
    }

    #[test]
    fn sum_constant_signed() {
        let array = ConstantArray::new(-5i64, 10).into_array();
        let result = sum(&array).unwrap();
        assert_eq!(result, (-50i64).into());
    }

    #[test]
    fn sum_constant_nullable_value() {
        let array = ConstantArray::new(Scalar::null(DType::Primitive(PType::U32, Nullable)), 10)
            .into_array();
        let result = sum(&array).unwrap();
        assert_eq!(result, Scalar::primitive(0u64, Nullable));
    }

    #[test]
    fn sum_constant_bool_false() {
        let array = ConstantArray::new(false, 10).into_array();
        let result = sum(&array).unwrap();
        assert_eq!(result, 0u64.into());
    }

    #[test]
    fn sum_constant_bool_true() {
        let array = ConstantArray::new(true, 10).into_array();
        let result = sum(&array).unwrap();
        assert_eq!(result, 10u64.into());
    }

    #[test]
    fn sum_constant_bool_null() {
        let array = ConstantArray::new(Scalar::null(DType::Bool(Nullable)), 10).into_array();
        let result = sum(&array).unwrap();
        assert_eq!(result, Scalar::primitive(0u64, Nullable));
    }

    #[test]
    fn sum_constant_decimal() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I64(100),
                decimal_dtype,
                Nullability::NonNullable,
            ),
            5,
        )
        .into_array();

        let result = sum(&array).unwrap();

        assert_eq!(
            result.as_decimal().decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(500)))
        );
        assert_eq!(result.dtype(), &Stat::Sum.dtype(array.dtype()).unwrap());
    }

    #[test]
    fn sum_constant_decimal_null() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = ConstantArray::new(Scalar::null(DType::Decimal(decimal_dtype, Nullable)), 10)
            .into_array();

        let result = sum(&array).unwrap();
        assert_eq!(
            result,
            Scalar::decimal(
                DecimalValue::I256(i256::ZERO),
                DecimalDType::new(20, 2),
                Nullable
            )
        );
    }

    #[test]
    fn sum_constant_decimal_large_value() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let array = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I64(999_999_999),
                decimal_dtype,
                Nullability::NonNullable,
            ),
            100,
        )
        .into_array();

        let result = sum(&array).unwrap();
        assert_eq!(
            result.as_decimal().decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(99_999_999_900)))
        );
    }

    #[test]
    fn sum_constant_float_non_multiply() {
        let acc = -2048669276050936500000000000f64;
        let array = ConstantArray::new(6.1811675e16f64, 25);
        let sum = sum_with_accumulator(&array.into_array(), &Scalar::primitive(acc, Nullable))
            .vortex_expect("operation should succeed in test");
        assert_eq!(
            f64::try_from(&sum).vortex_expect("operation should succeed in test"),
            -2048669274505644600000000000f64
        );
    }

    // -- Decimal array tests (migrated from decimal/compute/sum.rs) --

    #[test]
    fn sum_decimal_basic() {
        let decimal = DecimalArray::new(
            buffer![100i32, 200i32, 300i32],
            DecimalDType::new(4, 2),
            Validity::AllValid,
        );

        let result = sum(&decimal.into_array()).unwrap();

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(14, 2), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(600i32))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn sum_decimal_with_nulls() {
        let decimal = DecimalArray::new(
            buffer![100i32, 200i32, 300i32, 400i32],
            DecimalDType::new(4, 2),
            Validity::from_iter([true, false, true, true]),
        );

        let result = sum(&decimal.into_array()).unwrap();

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(14, 2), Nullable),
            Some(ScalarValue::from(DecimalValue::from(800i32))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn sum_decimal_negative_values() {
        let decimal = DecimalArray::new(
            buffer![100i32, -200i32, 300i32, -50i32],
            DecimalDType::new(4, 2),
            Validity::AllValid,
        );

        let result = sum(&decimal.into_array()).unwrap();

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(14, 2), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(150i32))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn sum_decimal_near_i32_max() {
        let near_max = i32::MAX - 1000;
        let decimal = DecimalArray::new(
            buffer![near_max, 500i32, 400i32],
            DecimalDType::new(10, 2),
            Validity::AllValid,
        );

        let result = sum(&decimal.into_array()).unwrap();

        let expected_sum = near_max as i64 + 500 + 400;
        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(20, 2), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(expected_sum))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn sum_decimal_large_i64_values() {
        let large_val = i64::MAX / 4;
        let decimal = DecimalArray::new(
            buffer![large_val, large_val, large_val, large_val + 1],
            DecimalDType::new(19, 0),
            Validity::AllValid,
        );

        let result = sum(&decimal.into_array()).unwrap();

        let expected_sum = (large_val as i128) * 4 + 1;
        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(29, 0), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(expected_sum))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn sum_decimal_preserves_scale() {
        let decimal = DecimalArray::new(
            buffer![12345i32, 67890i32, 11111i32],
            DecimalDType::new(6, 4),
            Validity::AllValid,
        );

        let result = sum(&decimal.into_array()).unwrap();

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(16, 4), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(91346i32))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn sum_decimal_single_value() {
        let decimal =
            DecimalArray::new(buffer![42i32], DecimalDType::new(3, 1), Validity::AllValid);

        let result = sum(&decimal.into_array()).unwrap();

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(13, 1), Nullability::NonNullable),
            Some(ScalarValue::from(DecimalValue::from(42i32))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn sum_decimal_all_nulls_except_one() {
        let decimal = DecimalArray::new(
            buffer![100i32, 200i32, 300i32, 400i32],
            DecimalDType::new(4, 2),
            Validity::from_iter([false, false, true, false]),
        );

        let result = sum(&decimal.into_array()).unwrap();

        let expected = Scalar::try_new(
            DType::Decimal(DecimalDType::new(14, 2), Nullable),
            Some(ScalarValue::from(DecimalValue::from(300i32))),
        )
        .unwrap();

        assert_eq!(result, expected);
    }

    #[test]
    fn sum_decimal_overflow_detection() {
        let max_val = i128::MAX / 2;
        let decimal = DecimalArray::new(
            buffer![max_val, max_val, max_val],
            DecimalDType::new(38, 0),
            Validity::AllValid,
        );

        let result = sum(&decimal.into_array()).unwrap();

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
    fn sum_decimal_i256_overflow() {
        let decimal_dtype = DecimalDType::new(76, 0);
        let decimal = DecimalArray::new(
            buffer![i256::MAX, i256::MAX, i256::MAX],
            decimal_dtype,
            Validity::AllValid,
        );

        assert_eq!(
            sum(&decimal.into_array()).vortex_expect("operation should succeed in test"),
            Scalar::null(DType::Decimal(decimal_dtype, Nullable))
        );
    }

    // -- Chunked array tests (migrated from chunked/compute/sum.rs) --

    #[test]
    fn sum_chunked_floats_with_nulls() {
        let chunk1 =
            PrimitiveArray::from_option_iter(vec![Some(1.5f64), None, Some(3.2), Some(4.8)]);
        let chunk2 = PrimitiveArray::from_option_iter(vec![Some(2.1f64), Some(5.7), None]);
        let chunk3 = PrimitiveArray::from_option_iter(vec![None, Some(1.0f64), Some(2.5), None]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(
            vec![
                chunk1.into_array(),
                chunk2.into_array(),
                chunk3.into_array(),
            ],
            dtype,
        )
        .unwrap();

        let result = sum(&chunked.into_array()).unwrap();
        assert_eq!(result.as_primitive().as_::<f64>(), Some(20.8));
    }

    #[test]
    fn sum_chunked_floats_all_nulls_is_zero() {
        let chunk1 = PrimitiveArray::from_option_iter::<f32, _>(vec![None, None, None]);
        let chunk2 = PrimitiveArray::from_option_iter::<f32, _>(vec![None, None]);
        let dtype = chunk1.dtype().clone();
        let chunked =
            ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype).unwrap();
        let result = sum(&chunked.into_array()).unwrap();
        assert_eq!(result, Scalar::primitive(0f64, Nullable));
    }

    #[test]
    fn sum_chunked_floats_empty_chunks() {
        let chunk1 = PrimitiveArray::from_option_iter(vec![Some(10.5f64), Some(20.3)]);
        let chunk2 = ConstantArray::new(Scalar::primitive(0f64, Nullable), 0);
        let chunk3 = PrimitiveArray::from_option_iter(vec![Some(5.2f64)]);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(
            vec![
                chunk1.into_array(),
                chunk2.into_array(),
                chunk3.into_array(),
            ],
            dtype,
        )
        .unwrap();

        let result = sum(&chunked.into_array()).unwrap();
        assert_eq!(result.as_primitive().as_::<f64>(), Some(36.0));
    }

    #[test]
    fn sum_chunked_int_almost_all_null() {
        let chunk1 = PrimitiveArray::from_option_iter::<u32, _>(vec![Some(1)]);
        let chunk2 = PrimitiveArray::from_option_iter::<u32, _>(vec![None]);
        let dtype = chunk1.dtype().clone();
        let chunked =
            ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype).unwrap();

        let result = sum(&chunked.into_array()).unwrap();
        assert_eq!(result.as_primitive().as_::<u64>(), Some(1));
    }

    #[test]
    fn sum_chunked_decimals() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let chunk1 = DecimalArray::new(
            buffer![100i32, 100i32, 100i32, 100i32, 100i32],
            decimal_dtype,
            Validity::AllValid,
        );
        let chunk2 = DecimalArray::new(
            buffer![200i32, 200i32, 200i32],
            decimal_dtype,
            Validity::AllValid,
        );
        let chunk3 = DecimalArray::new(buffer![300i32, 300i32], decimal_dtype, Validity::AllValid);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(
            vec![
                chunk1.into_array(),
                chunk2.into_array(),
                chunk3.into_array(),
            ],
            dtype,
        )
        .unwrap();

        let result = sum(&chunked.into_array()).unwrap();
        let decimal_result = result.as_decimal();
        assert_eq!(
            decimal_result.decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(1700)))
        );
    }

    #[test]
    fn sum_chunked_decimals_with_nulls() {
        let decimal_dtype = DecimalDType::new(10, 2);
        let chunk1 = DecimalArray::new(
            buffer![100i32, 100i32, 100i32],
            decimal_dtype,
            Validity::AllValid,
        );
        let chunk2 = DecimalArray::new(
            buffer![0i32, 0i32],
            decimal_dtype,
            Validity::from_iter([false, false]),
        );
        let chunk3 = DecimalArray::new(buffer![200i32, 200i32], decimal_dtype, Validity::AllValid);
        let dtype = chunk1.dtype().clone();
        let chunked = ChunkedArray::try_new(
            vec![
                chunk1.into_array(),
                chunk2.into_array(),
                chunk3.into_array(),
            ],
            dtype,
        )
        .unwrap();

        let result = sum(&chunked.into_array()).unwrap();
        let decimal_result = result.as_decimal();
        assert_eq!(
            decimal_result.decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(700)))
        );
    }

    #[test]
    fn sum_chunked_decimals_large() {
        let decimal_dtype = DecimalDType::new(3, 0);
        let chunk1 = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I16(500),
                decimal_dtype,
                Nullability::NonNullable,
            ),
            1,
        );
        let chunk2 = ConstantArray::new(
            Scalar::decimal(
                DecimalValue::I16(600),
                decimal_dtype,
                Nullability::NonNullable,
            ),
            1,
        );
        let dtype = chunk1.dtype().clone();
        let chunked =
            ChunkedArray::try_new(vec![chunk1.into_array(), chunk2.into_array()], dtype).unwrap();

        let result = sum(&chunked.into_array()).unwrap();
        let decimal_result = result.as_decimal();
        assert_eq!(
            decimal_result.decimal_value(),
            Some(DecimalValue::I256(i256::from_i128(1100)))
        );
        assert_eq!(
            result.dtype(),
            &DType::Decimal(DecimalDType::new(13, 0), Nullable)
        );
    }
}
