// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::result_decimal_dtype;
use crate::ArrayRef;
use crate::Columnar;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::VortexSessionExecute;
use crate::array_session;
use crate::arrays::ConstantArray;
use crate::arrays::DecimalArray;
use crate::arrays::PrimitiveArray;
use crate::assert_arrays_eq;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::DecimalType;
use crate::dtype::NativeDecimalType;
use crate::dtype::Nullability;
use crate::dtype::i256;
use crate::scalar::DecimalValue;
use crate::scalar::NumericOperator;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::operators::Operator;
use crate::validity::Validity;

fn sub_scalar(array: &ArrayRef, scalar: impl Into<Scalar>) -> VortexResult<ArrayRef> {
    array
        .binary(
            ConstantArray::new(scalar, array.len()).into_array(),
            Operator::Sub,
        )
        .and_then(|a| a.execute::<RecursiveCanonical>(&mut array_session().create_execution_ctx()))
        .map(|a| a.0.into_array())
}

#[test]
fn test_scalar_subtract_unsigned() {
    let mut ctx = array_session().create_execution_ctx();
    let values = buffer![1u16, 2, 3].into_array();
    let result = sub_scalar(&values, 1u16).unwrap();
    assert_arrays_eq!(result, PrimitiveArray::from_iter([0u16, 1, 2]), &mut ctx);
}

#[test]
fn test_scalar_subtract_signed() {
    let mut ctx = array_session().create_execution_ctx();
    let values = buffer![1i64, 2, 3].into_array();
    let result = sub_scalar(&values, -1i64).unwrap();
    assert_arrays_eq!(result, PrimitiveArray::from_iter([2i64, 3, 4]), &mut ctx);
}

#[test]
fn test_scalar_subtract_nullable() {
    let mut ctx = array_session().create_execution_ctx();
    let values = PrimitiveArray::from_option_iter([Some(1u16), Some(2), None, Some(3)]);
    let result = sub_scalar(&values.into_array(), Some(1u16)).unwrap();
    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(0u16), Some(1), None, Some(2)]),
        &mut ctx
    );
}

#[test]
fn test_scalar_subtract_float() {
    let mut ctx = array_session().create_execution_ctx();
    let values = buffer![1.0f64, 2.0, 3.0].into_array();
    let result = sub_scalar(&values, -1f64).unwrap();
    assert_arrays_eq!(
        result,
        PrimitiveArray::from_iter([2.0f64, 3.0, 4.0]),
        &mut ctx
    );
}

#[test]
fn test_scalar_subtract_float_underflow_is_ok() {
    let values = buffer![f32::MIN, 2.0, 3.0].into_array();
    let _results = sub_scalar(&values, 1.0f32).unwrap();
    let _results = sub_scalar(&values, f32::MAX).unwrap();
}

#[test]
fn test_float_divide_by_zero_is_ok() {
    let mut ctx = array_session().create_execution_ctx();
    let values = buffer![1.0f64, -1.0].into_array();
    let result = values
        .binary(
            ConstantArray::new(0.0f64, values.len()).into_array(),
            Operator::Div,
        )
        .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()))
        .unwrap();

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_iter([f64::INFINITY, f64::NEG_INFINITY]),
        &mut ctx
    );
}

#[test]
fn test_integer_overflow_errors() {
    let values = buffer![u8::MAX].into_array();
    let result = values
        .binary(
            ConstantArray::new(1u8, values.len()).into_array(),
            Operator::Add,
        )
        .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()));

    assert!(result.is_err());
}

#[test]
fn test_integer_divide_by_zero_errors() {
    let values = buffer![1i32].into_array();
    let result = values
        .binary(
            ConstantArray::new(0i32, values.len()).into_array(),
            Operator::Div,
        )
        .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()));

    assert!(result.is_err());
}

#[test]
fn test_integer_divide_overflow_errors() {
    let values = buffer![i64::MIN].into_array();
    let result = values
        .binary(
            ConstantArray::new(-1i64, values.len()).into_array(),
            Operator::Div,
        )
        .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()));

    assert!(result.is_err());
}

#[test]
fn test_integer_divide_errors_ignore_null_lanes() {
    let mut ctx = array_session().create_execution_ctx();
    let lhs =
        PrimitiveArray::new(buffer![10i32, 10], Validity::from_iter([false, true])).into_array();
    let rhs = buffer![0i32, 2].into_array();
    let result = lhs
        .binary(rhs, Operator::Div)
        .and_then(|a| a.execute::<RecursiveCanonical>(&mut array_session().create_execution_ctx()))
        .map(|a| a.0.into_array())
        .unwrap();

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([None, Some(5i32)]),
        &mut ctx
    );
}

#[test]
fn test_integer_errors_ignore_null_lanes() {
    let mut ctx = array_session().create_execution_ctx();
    let values =
        PrimitiveArray::new(buffer![u8::MAX, 1], Validity::from_iter([false, true])).into_array();
    let result = values
        .binary(
            ConstantArray::new(1u8, values.len()).into_array(),
            Operator::Add,
        )
        .and_then(|a| a.execute::<RecursiveCanonical>(&mut array_session().create_execution_ctx()))
        .map(|a| a.0.into_array())
        .unwrap();

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([None, Some(2u8)]),
        &mut ctx
    );
}

#[test]
fn test_integer_array_array_errors_on_valid_lanes() {
    let lhs = PrimitiveArray::new(
        buffer![u8::MAX, 1, u8::MAX],
        Validity::from_iter([false, true, true]),
    )
    .into_array();
    let rhs = buffer![1u8, 1, 1].into_array();
    let result = lhs
        .binary(rhs, Operator::Add)
        .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()));

    assert!(result.is_err());
}

#[test]
fn test_present_nullable_constant_preserves_nullable_output() {
    let mut ctx = array_session().create_execution_ctx();
    let values = buffer![1u8, 2].into_array();
    let result = values
        .binary(
            ConstantArray::new(Some(1u8), values.len()).into_array(),
            Operator::Add,
        )
        .and_then(|a| a.execute::<PrimitiveArray>(&mut array_session().create_execution_ctx()))
        .unwrap();

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([Some(2u8), Some(3)]),
        &mut ctx
    );
}

#[test]
fn test_empty_primitive_constants_do_not_evaluate() -> VortexResult<()> {
    let lhs = ConstantArray::new(u8::MAX, 0).into_array();
    let rhs = ConstantArray::new(1u8, 0).into_array();

    let result = lhs
        .binary(rhs, Operator::Add)?
        .execute::<RecursiveCanonical>(&mut array_session().create_execution_ctx())?;

    assert!(result.0.is_empty());
    Ok(())
}

// -- Decimal arithmetic --

fn decimal_binary(lhs: ArrayRef, rhs: ArrayRef, op: Operator) -> VortexResult<ArrayRef> {
    lhs.binary(rhs, op)
        .and_then(|a| a.execute::<RecursiveCanonical>(&mut array_session().create_execution_ctx()))
        .map(|a| a.0.into_array())
}

fn decimal_constant(value: impl Into<DecimalValue>, dtype: DecimalDType, len: usize) -> ArrayRef {
    ConstantArray::new(
        Scalar::decimal(value.into(), dtype, Nullability::NonNullable),
        len,
    )
    .into_array()
}

#[rstest]
#[case::add(NumericOperator::Add, [150i64, 225], [1050i64, 1225])]
#[case::sub(NumericOperator::Sub, [150i64, 225], [750i64, 775])]
fn test_decimal_array_array(
    #[case] op: NumericOperator,
    #[case] rhs: [i64; 2],
    #[case] expected: [i64; 2],
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let result_dtype = result_decimal_dtype(dtype, op)?;
    let lhs = DecimalArray::from_iter::<i64, _>([900, 1000], dtype).into_array();
    let rhs = DecimalArray::from_iter::<i64, _>(rhs, dtype).into_array();

    let result = decimal_binary(lhs, rhs, op.into())?;
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i64, _>(expected, result_dtype),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_decimal_mixed_storage_widths() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let lhs = DecimalArray::from_iter::<i32, _>([100, 250], dtype).into_array();
    let rhs = DecimalArray::from_iter::<i128, _>([200, 250], dtype).into_array();

    let result = decimal_binary(lhs, rhs, Operator::Add)?;
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i64, _>(
            [300, 500],
            result_decimal_dtype(dtype, NumericOperator::Add)?,
        ),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_decimal_nullable_lanes() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let lhs =
        DecimalArray::from_option_iter::<i64, _>([Some(100), None, Some(300)], dtype).into_array();
    let rhs = DecimalArray::from_iter::<i64, _>([50, 50, 50], dtype).into_array();

    let result = decimal_binary(lhs, rhs, Operator::Add)?;
    assert_arrays_eq!(
        result,
        DecimalArray::from_option_iter::<i64, _>(
            [Some(150), None, Some(350)],
            result_decimal_dtype(dtype, NumericOperator::Add)?,
        ),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_decimal_max_precision_overflow_on_valid_lane_errors() {
    let dtype = DecimalDType::new(76, 0);
    let max = <i256 as NativeDecimalType>::MAX_BY_PRECISION[76];
    let lhs = DecimalArray::from_iter::<i256, _>([max], dtype).into_array();
    let rhs = DecimalArray::from_iter::<i256, _>([i256::from_i128(1)], dtype).into_array();

    assert!(decimal_binary(lhs, rhs, Operator::Add).is_err());
}

#[test]
fn test_decimal_value_outside_working_width_errors() {
    let dtype = DecimalDType::new(2, 0);
    let value = i256::from_i128(1_000_000);
    let lhs = DecimalArray::new(buffer![value], dtype, Validity::NonNullable).into_array();
    let rhs = DecimalArray::new(buffer![value], dtype, Validity::NonNullable).into_array();

    assert!(decimal_binary(lhs, rhs, Operator::Add).is_err());
}

#[test]
fn test_decimal_value_outside_working_width_on_null_lane_ignored() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(2, 0);
    let result_dtype = result_decimal_dtype(dtype, NumericOperator::Add)?;
    let lhs = DecimalArray::new(
        buffer![i256::from_i128(1_000_000), i256::from_i128(1)],
        dtype,
        Validity::from_iter([false, true]),
    )
    .into_array();
    let rhs = decimal_constant(i256::from_i128(1), dtype, 2);

    let result = decimal_binary(lhs, rhs, Operator::Add)?;
    assert_arrays_eq!(
        result,
        DecimalArray::from_option_iter::<i16, _>([None, Some(2)], result_dtype),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_decimal_overflow_on_null_lane_ignored() {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(76, 0);
    let max = <i256 as NativeDecimalType>::MAX_BY_PRECISION[76];
    let one = i256::from_i128(1);
    // The null lane holds the maximum value, so adding one overflows there but is ignored.
    let lhs = DecimalArray::new(buffer![max, one], dtype, Validity::from_iter([false, true]))
        .into_array();
    let rhs = decimal_constant(one, dtype, 2);

    let result = decimal_binary(lhs, rhs, Operator::Add).unwrap();
    assert_arrays_eq!(
        result,
        DecimalArray::from_option_iter::<i256, _>([None, Some(i256::from_i128(2))], dtype,),
        &mut ctx
    );
}

#[test]
fn test_decimal_add_reserves_carry_digit() {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(2, 0);
    let lhs = DecimalArray::from_iter::<i8, _>([99], dtype).into_array();
    let rhs = DecimalArray::from_iter::<i8, _>([99], dtype).into_array();

    let result = decimal_binary(lhs, rhs, Operator::Add).unwrap();
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i16, _>([198], DecimalDType::new(3, 0)),
        &mut ctx
    );
}

#[rstest]
#[case::precision_1(1, 2)]
#[case::precision_17(17, 18)]
#[case::precision_37(37, 38)]
// 39 digits do not fit an i128, so the carry digit is dropped rather than widening to i256.
#[case::precision_38(38, 38)]
#[case::precision_39(39, 40)]
#[case::precision_75(75, 76)]
#[case::precision_76(76, 76)]
fn test_decimal_add_sub_result_precision(
    #[case] precision: u8,
    #[case] expected: u8,
    #[values(NumericOperator::Add, NumericOperator::Sub)] op: NumericOperator,
) -> VortexResult<()> {
    let result = result_decimal_dtype(DecimalDType::new(precision, 0), op)?;
    assert_eq!(result.precision(), expected);
    Ok(())
}

#[test]
fn test_decimal_precision_38_saturation_reports_overflow() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(38, 0);
    let max = <i128 as NativeDecimalType>::MAX_BY_PRECISION[38];

    // A sum that still fits 38 digits is exact and stays at the i128 working width.
    let result = decimal_binary(
        DecimalArray::from_iter::<i128, _>([max - 1], dtype).into_array(),
        DecimalArray::from_iter::<i128, _>([1], dtype).into_array(),
        Operator::Add,
    )?;
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i128, _>([max], dtype),
        &mut ctx
    );

    // A sum needing the 39th digit is an overflow error, not a silent widening to i256.
    assert!(
        decimal_binary(
            DecimalArray::from_iter::<i128, _>([max], dtype).into_array(),
            DecimalArray::from_iter::<i128, _>([1], dtype).into_array(),
            Operator::Add,
        )
        .is_err()
    );
    Ok(())
}

/// Every lane must agree with exact 256-bit arithmetic bounded by the result precision: an
/// in-range pair produces the exact value, an out-of-range pair is an error. Covers each working
/// width, including the two precisions where the result precision saturates.
#[rstest]
#[case::precision_2(2)]
#[case::precision_4(4)]
#[case::precision_9(9)]
#[case::precision_18(18)]
#[case::precision_38(38)]
#[case::precision_39(39)]
#[case::precision_76(76)]
fn test_decimal_add_sub_matches_exact_arithmetic(
    #[case] precision: u8,
    #[values(NumericOperator::Add, NumericOperator::Sub)] op: NumericOperator,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(precision, 0);
    let result_dtype = result_decimal_dtype(dtype, op)?;
    let result_precision = result_dtype.precision() as usize;
    let lower = <i256 as NativeDecimalType>::MIN_BY_PRECISION[result_precision];
    let upper = <i256 as NativeDecimalType>::MAX_BY_PRECISION[result_precision];

    let max = <i256 as NativeDecimalType>::MAX_BY_PRECISION[precision as usize];
    let half = max / i256::from_i128(2);
    let operands = [i256::ZERO, i256::ONE, -i256::ONE, max, -max, half, -half];

    let mut in_range = Vec::new();
    for lhs in operands {
        for rhs in operands {
            // Both operands are in precision, so the exact result cannot itself overflow an i256.
            let exact = match op {
                NumericOperator::Add => lhs + rhs,
                _ => lhs - rhs,
            };
            if lower <= exact && exact <= upper {
                in_range.push((lhs, rhs, exact));
                continue;
            }
            assert!(
                decimal_binary(
                    DecimalArray::from_iter::<i256, _>([lhs], dtype).into_array(),
                    DecimalArray::from_iter::<i256, _>([rhs], dtype).into_array(),
                    op.into(),
                )
                .is_err(),
                "{lhs} {op} {rhs} is outside decimal({result_precision}, 0) and must error"
            );
        }
    }

    let result = decimal_binary(
        DecimalArray::from_iter::<i256, _>(in_range.iter().map(|(lhs, ..)| *lhs), dtype)
            .into_array(),
        DecimalArray::from_iter::<i256, _>(in_range.iter().map(|(_, rhs, _)| *rhs), dtype)
            .into_array(),
        op.into(),
    )?;
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i256, _>(
            in_range.iter().map(|(_, _, exact)| *exact),
            result_dtype
        ),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_decimal_working_width_overflow_wrapping_into_precision_errors() {
    // Out-of-precision inputs whose sum wraps the i128 working width back into the in-precision
    // range: `i128::MAX + i128::MAX` wraps to -2. The bounds check alone would accept that, so the
    // overflow check has to reject it first.
    let dtype = DecimalDType::new(38, 0);
    let lhs = DecimalArray::from_iter::<i128, _>([i128::MAX], dtype).into_array();
    let rhs = DecimalArray::from_iter::<i128, _>([i128::MAX], dtype).into_array();

    assert!(decimal_binary(lhs, rhs, Operator::Add).is_err());
}

#[rstest]
#[case::precision_2(
    DecimalArray::from_iter::<i8, _>([10, 20], DecimalDType::new(2, 0)),
    DecimalType::I16,
)]
#[case::precision_18(
    DecimalArray::from_iter::<i64, _>([10, 20], DecimalDType::new(18, 0)),
    DecimalType::I128,
)]
#[case::precision_37(
    DecimalArray::from_iter::<i128, _>([10, 20], DecimalDType::new(37, 0)),
    DecimalType::I128,
)]
// The carry digit saturates at 38 rather than promoting the result to an i256 working width.
#[case::precision_38(
    DecimalArray::from_iter::<i128, _>([10, 20], DecimalDType::new(38, 0)),
    DecimalType::I128,
)]
#[case::precision_76(
    DecimalArray::from_iter::<i256, _>(
        [i256::from_i128(10), i256::from_i128(20)],
        DecimalDType::new(76, 0),
    ),
    DecimalType::I256,
)]
fn test_decimal_result_uses_widened_logical_storage_width(
    #[case] values: DecimalArray,
    #[case] expected_type: DecimalType,
) -> VortexResult<()> {
    let expected_dtype = result_decimal_dtype(values.decimal_dtype(), NumericOperator::Add)?;
    let result = decimal_binary(
        values.clone().into_array(),
        values.into_array(),
        Operator::Add,
    )?
    .execute::<DecimalArray>(&mut array_session().create_execution_ctx())?;

    assert_eq!(result.values_type(), expected_type);
    assert_eq!(result.decimal_dtype(), expected_dtype);
    Ok(())
}

#[test]
fn test_decimal_precision_76_boundary() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(76, 0);
    let max = <i256 as NativeDecimalType>::MAX_BY_PRECISION[76];
    let zero = i256::from_i128(0);
    let one = i256::from_i128(1);

    let result = decimal_binary(
        DecimalArray::from_iter::<i256, _>([max], dtype).into_array(),
        DecimalArray::from_iter::<i256, _>([zero], dtype).into_array(),
        Operator::Add,
    )?;
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i256, _>([max], dtype),
        &mut ctx
    );

    let overflow = decimal_binary(
        DecimalArray::from_iter::<i256, _>([max], dtype).into_array(),
        DecimalArray::from_iter::<i256, _>([one], dtype).into_array(),
        Operator::Add,
    );
    assert!(overflow.is_err());
    Ok(())
}

#[test]
fn test_decimal_empty_constants_do_not_evaluate() -> VortexResult<()> {
    let dtype = DecimalDType::new(76, 0);
    let max = <i256 as NativeDecimalType>::MAX_BY_PRECISION[76];
    let lhs = decimal_constant(max, dtype, 0);
    let rhs = decimal_constant(i256::from_i128(1), dtype, 0);

    let result = decimal_binary(lhs, rhs, Operator::Add)?;

    assert!(result.is_empty());
    assert_eq!(
        result.dtype(),
        &DType::Decimal(dtype, Nullability::NonNullable)
    );
    Ok(())
}

#[test]
fn test_decimal_constant_lhs_non_commutative() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let lhs = decimal_constant(1000i64, dtype, 2);
    let rhs = DecimalArray::from_iter::<i64, _>([250, 400], dtype).into_array();

    let result = decimal_binary(lhs, rhs, Operator::Sub)?;
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i64, _>(
            [750, 600],
            result_decimal_dtype(dtype, NumericOperator::Sub)?,
        ),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_decimal_nullable_constant_preserves_nullable_output() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let values = DecimalArray::from_iter::<i64, _>([100, 200], dtype).into_array();
    let constant = ConstantArray::new(
        Scalar::decimal(DecimalValue::from(50i64), dtype, Nullability::Nullable),
        2,
    )
    .into_array();

    let result = decimal_binary(values, constant, Operator::Add)?;
    assert_arrays_eq!(
        result,
        DecimalArray::from_option_iter::<i64, _>(
            [Some(150), Some(250)],
            result_decimal_dtype(dtype, NumericOperator::Add)?,
        ),
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case::null_lhs(true)]
#[case::null_rhs(false)]
fn test_decimal_null_constant_yields_all_null(#[case] null_lhs: bool) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let values = DecimalArray::from_iter::<i64, _>([100, 200], dtype).into_array();
    let null_constant = ConstantArray::new(
        Scalar::null(DType::Decimal(dtype, Nullability::Nullable)),
        2,
    )
    .into_array();
    let (lhs, rhs) = if null_lhs {
        (null_constant, values)
    } else {
        (values, null_constant)
    };

    let result = lhs
        .binary(rhs, Operator::Add)?
        .execute::<Columnar>(&mut ctx)?;
    assert!(matches!(&result, Columnar::Constant(_)));
    assert_arrays_eq!(
        result.into_array(),
        DecimalArray::from_option_iter::<i64, _>(
            [None, None],
            result_decimal_dtype(dtype, NumericOperator::Add)?,
        ),
        &mut ctx
    );
    Ok(())
}

/// A constant stored in a wider variant than the array storage participates through the widened
/// working type.
#[test]
fn test_decimal_constant_wider_than_array_storage() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(20, 0);
    let values = DecimalArray::from_iter::<i8, _>([1, 2], dtype).into_array();
    let constant = decimal_constant(10_000_000_000i64, dtype, 2);

    let result = decimal_binary(values, constant, Operator::Add)?;
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i64, _>(
            [10_000_000_001, 10_000_000_002],
            result_decimal_dtype(dtype, NumericOperator::Add)?,
        ),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_decimal_empty() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let empty = DecimalArray::from_iter::<i64, _>([], dtype).into_array();

    let result = decimal_binary(empty.clone(), empty, Operator::Add)?;
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i64, _>([], result_decimal_dtype(dtype, NumericOperator::Add)?,),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_decimal_constant_constant_folds() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let lhs = decimal_constant(150i64, dtype, 3);
    let rhs = decimal_constant(50i64, dtype, 3);

    let result = decimal_binary(lhs, rhs, Operator::Add)?;
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i64, _>(
            [200, 200, 200],
            result_decimal_dtype(dtype, NumericOperator::Add)?,
        ),
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case::mul(Operator::Mul)]
#[case::div(Operator::Div)]
fn test_decimal_mul_div_unsupported(#[case] op: Operator) {
    let dtype = DecimalDType::new(10, 2);
    let values = DecimalArray::from_iter::<i64, _>([100], dtype).into_array();

    assert!(decimal_binary(values.clone(), values, op).is_err());
}
