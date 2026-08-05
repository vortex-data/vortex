// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use rstest::rstest;
use vortex_buffer::Buffer;
use vortex_buffer::buffer;
use vortex_error::VortexResult;

use super::numeric_op_result_decimal_dtype as result_decimal_dtype;
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
use crate::dtype::NativePType;
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

/// Multiply two non-nullable lanes of `lhs` by two of `rhs`, expecting `Some(product)` where the
/// product fits and `None` where the checked kernel must report overflow.
#[track_caller]
fn assert_multiply<T: NativePType>(lhs: T, rhs: T, expected: Option<T>) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let result = PrimitiveArray::from_iter([lhs, lhs])
        .into_array()
        .binary(
            PrimitiveArray::from_iter([rhs, rhs]).into_array(),
            Operator::Mul,
        )?
        .execute::<PrimitiveArray>(&mut ctx);

    let Some(product) = expected else {
        assert!(result.is_err(), "{lhs:?} * {rhs:?} must report overflow");
        return Ok(());
    };

    assert_arrays_eq!(
        result?,
        PrimitiveArray::from_iter([product, product]),
        &mut ctx
    );

    Ok(())
}

/// Multiplication derives overflow from the bits the narrow product discards rather than from a
/// comparison, and it does so by a different formula per width, so each boundary of each formula is
/// worth pinning. The 8 and 32-bit signed cases cover the two-sided range check, the 64-bit signed
/// ones the sign-extension XOR, and the unsigned ones the high-half shift.
#[rstest]
#[case::i8_fits(100i8, 1, Some(100))]
#[case::i8_overflows(100i8, 2, None)]
#[case::i8_min_times_minus_one(i8::MIN,     -1,       None)]
#[case::i32_fits(             1i32 << 15,   1 << 15,  Some(1 << 30))]
#[case::i32_min_times_minus_one(i32::MIN,   -1,       None)]
#[case::i64_min_times_one(i64::MIN, 1, Some(i64::MIN))]
#[case::i64_minus_one_squared(-1i64,        -1,       Some(1))]
#[case::i64_max_times_one(i64::MAX, 1, Some(i64::MAX))]
#[case::i64_negative_product( -5i64,        3,        Some(-15))]
#[case::i64_zero(0i64, i64::MIN, Some(0))]
#[case::i64_min_times_minus_one(i64::MIN,   -1,       None)]
#[case::i64_max_times_two(i64::MAX, 2, None)]
#[case::i64_min_times_two(i64::MIN, 2, None)]
#[case::u8_fits(200u8, 1, Some(200))]
#[case::u8_overflows(200u8, 2, None)]
#[case::u16_boundary(65535u16, 1, Some(65535))]
#[case::u32_halves(           1u32 << 16,   1 << 15,  Some(1 << 31))]
#[case::u32_overflows(        1u32 << 16,   1 << 16,  None)]
#[case::u64_max(u64::MAX, 1, Some(u64::MAX))]
#[case::u64_overflows(u64::MAX, 2, None)]
fn test_multiply_overflow_boundaries<T: NativePType>(
    #[case] lhs: T,
    #[case] rhs: T,
    #[case] expected: Option<T>,
) -> VortexResult<()> {
    assert_multiply(lhs, rhs, expected)
}

/// An overflowing multiply behind a null row stays invisible, whichever width of evidence the lane
/// reports: `bool` for narrow signed, the operand itself for unsigned, and `u64` for 64-bit.
#[rstest]
#[case::signed_8(i8::MAX, 3, 9)]
#[case::unsigned_32(u32::MAX, 7, 49)]
#[case::signed_64(i64::MAX, 2, 4)]
fn test_multiply_overflow_on_null_lane_ignored<T: NativePType>(
    #[case] overflowing: T,
    #[case] rhs: T,
    #[case] expected: T,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let lhs = PrimitiveArray::new(
        Buffer::copy_from([overflowing, rhs]),
        Validity::from_iter([false, true]),
    )
    .into_array();

    let result = lhs
        .binary(
            PrimitiveArray::from_iter([rhs, rhs]).into_array(),
            Operator::Mul,
        )?
        .execute::<RecursiveCanonical>(&mut ctx)?
        .0
        .into_array();

    assert_arrays_eq!(
        result,
        PrimitiveArray::from_option_iter([None, Some(expected)]),
        &mut ctx
    );

    Ok(())
}

/// The hot pass OR-reduces evidence across whole 64-lane chunks before anything looks at it, so an
/// overflow in a late chunk must still be caught, and must still be suppressed when its lane is
/// null. Every other test here fits in a single chunk and cannot show either.
#[rstest]
#[case::reported(true)]
#[case::suppressed_by_null(false)]
fn test_multiply_overflow_survives_chunk_reduction(
    #[case] lane_is_valid: bool,
) -> VortexResult<()> {
    const LEN: u32 = 1000;
    const OVERFLOW_AT: u32 = 700;

    let mut ctx = array_session().create_execution_ctx();
    let mut lhs: Vec<u32> = (0..LEN).map(|i| i % 100 + 1).collect();
    lhs[OVERFLOW_AT as usize] = u32::MAX;

    let validity = Validity::from_iter((0..LEN).map(|i| i != OVERFLOW_AT || lane_is_valid));
    let result = PrimitiveArray::new(Buffer::copy_from(&lhs), validity)
        .into_array()
        .binary(
            PrimitiveArray::from_iter(vec![3u32; LEN as usize]).into_array(),
            Operator::Mul,
        )?
        .execute::<RecursiveCanonical>(&mut ctx);

    assert_eq!(result.is_err(), lane_is_valid);

    Ok(())
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
#[case::mul(NumericOperator::Mul, [150i64, 225], [135_000i64, 225_000])]
#[case::div(
    NumericOperator::Div,
    [150i64, 225],
    [6_000_000i64, 4_444_444]
)]
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
fn test_decimal_div_negative_result_scale() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    // A negative result scale scales the divisor rather than the dividend: 5e12 / 2e8 is 25_000,
    // truncated to 2 at the decimal(6, -4) result scale. The i64 working width is also wider than
    // the i32 result storage, so this narrows on the way out.
    let dtype = DecimalDType::new(10, -8);
    let lhs = DecimalArray::from_iter::<i64, _>([50_000], dtype).into_array();
    let rhs = DecimalArray::from_iter::<i64, _>([2], dtype).into_array();

    let result = decimal_binary(lhs, rhs, Operator::Div)?;
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i32, _>([2], DecimalDType::new(6, -4)),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_decimal_mul_value_outside_precision_errors() {
    // `DecimalArray::new` does not validate stored values against the declared precision, so Mul
    // cannot assume its inputs are in-precision: 500 * 500 is 250_000, well past the 99_999 that
    // the decimal(5, 0) result can represent.
    let dtype = DecimalDType::new(2, 0);
    let value = i256::from_i128(500);
    let lhs = DecimalArray::new(buffer![value], dtype, Validity::NonNullable).into_array();
    let rhs = DecimalArray::new(buffer![value], dtype, Validity::NonNullable).into_array();

    assert!(decimal_binary(lhs, rhs, Operator::Mul).is_err());
}

#[test]
fn test_decimal_mul_value_outside_working_width_errors() {
    // 50_000 * 50_000 overflows the i32 working width chosen for a decimal(5, 0) result, which
    // an unchecked multiply would wrap in release and panic on in debug.
    let dtype = DecimalDType::new(2, 0);
    let value = i256::from_i128(50_000);
    let lhs = DecimalArray::new(buffer![value], dtype, Validity::NonNullable).into_array();
    let rhs = DecimalArray::new(buffer![value], dtype, Validity::NonNullable).into_array();

    assert!(decimal_binary(lhs, rhs, Operator::Mul).is_err());
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
fn test_decimal_divide_by_zero_on_null_lane_ignored() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let lhs = DecimalArray::new(
        buffer![100i64, 1_000],
        dtype,
        Validity::from_iter([false, true]),
    )
    .into_array();
    let rhs = DecimalArray::from_iter::<i64, _>([0, 200], dtype).into_array();

    let result = decimal_binary(lhs, rhs, Operator::Div)?;
    assert_arrays_eq!(
        result,
        DecimalArray::from_option_iter::<i64, _>(
            [None, Some(5_000_000)],
            result_decimal_dtype(dtype, NumericOperator::Div)?,
        ),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_decimal_divide_by_zero_on_valid_lane_errors() {
    let dtype = DecimalDType::new(10, 2);
    let lhs = DecimalArray::from_iter::<i64, _>([100], dtype).into_array();
    let rhs = DecimalArray::from_iter::<i64, _>([0], dtype).into_array();

    assert!(decimal_binary(lhs, rhs, Operator::Div).is_err());
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

#[test]
fn test_decimal_mul_widens_before_multiplying() -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(38, 0);
    let max = <i128 as NativeDecimalType>::MAX_BY_PRECISION[38];
    let widened_max = i256::from_i128(max);
    let result_dtype = result_decimal_dtype(dtype, NumericOperator::Mul)?;

    let result = decimal_binary(
        DecimalArray::from_iter::<i128, _>([max], dtype).into_array(),
        DecimalArray::from_iter::<i128, _>([max], dtype).into_array(),
        Operator::Mul,
    )?;
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i256, _>([widened_max * widened_max], result_dtype),
        &mut ctx
    );
    Ok(())
}

#[test]
fn test_decimal_mul_above_result_precision_errors() {
    let dtype = DecimalDType::new(39, 0);
    let one = i256::from_i128(1);
    let ten_to_38 = <i256 as NativeDecimalType>::MAX_BY_PRECISION[38] + one;
    let value = ten_to_38 * i256::from_i128(2);
    let product = value * value;

    // The product fits the native i256 width but not the capped precision-76 result.
    assert!(product > <i256 as NativeDecimalType>::MAX_BY_PRECISION[76]);
    assert!(
        decimal_binary(
            DecimalArray::from_iter::<i256, _>([value], dtype).into_array(),
            DecimalArray::from_iter::<i256, _>([value], dtype).into_array(),
            Operator::Mul,
        )
        .is_err()
    );
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
#[case::precision_38(
    DecimalArray::from_iter::<i128, _>([10, 20], DecimalDType::new(38, 0)),
    DecimalType::I256,
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
#[case::add_null_lhs(true, NumericOperator::Add)]
#[case::add_null_rhs(false, NumericOperator::Add)]
#[case::mul_null_lhs(true, NumericOperator::Mul)]
#[case::mul_null_rhs(false, NumericOperator::Mul)]
#[case::div_null_lhs(true, NumericOperator::Div)]
#[case::div_null_rhs(false, NumericOperator::Div)]
fn test_decimal_null_constant_yields_all_null(
    #[case] null_lhs: bool,
    #[case] op: NumericOperator,
) -> VortexResult<()> {
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

    let result = lhs.binary(rhs, op.into())?.execute::<Columnar>(&mut ctx)?;
    assert!(matches!(&result, Columnar::Constant(_)));
    assert_arrays_eq!(
        result.into_array(),
        DecimalArray::from_option_iter::<i256, _>([None, None], result_decimal_dtype(dtype, op)?,),
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

#[rstest]
#[case::add(NumericOperator::Add)]
#[case::sub(NumericOperator::Sub)]
#[case::mul(NumericOperator::Mul)]
#[case::div(NumericOperator::Div)]
fn test_decimal_empty(#[case] op: NumericOperator) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let empty = DecimalArray::from_iter::<i64, _>([], dtype).into_array();

    let result = decimal_binary(empty.clone(), empty, op.into())?;
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i256, _>([], result_decimal_dtype(dtype, op)?,),
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case::add(NumericOperator::Add, 200)]
#[case::sub(NumericOperator::Sub, 100)]
#[case::mul(NumericOperator::Mul, 7_500)]
#[case::div(NumericOperator::Div, 3_000_000)]
fn test_decimal_constant_constant_folds(
    #[case] op: NumericOperator,
    #[case] expected: i128,
) -> VortexResult<()> {
    let mut ctx = array_session().create_execution_ctx();
    let dtype = DecimalDType::new(10, 2);
    let lhs = decimal_constant(150i64, dtype, 3);
    let rhs = decimal_constant(50i64, dtype, 3);

    let result = decimal_binary(lhs, rhs, op.into())?;
    assert_arrays_eq!(
        result,
        DecimalArray::from_iter::<i256, _>(
            [i256::from_i128(expected); 3],
            result_decimal_dtype(dtype, op)?,
        ),
        &mut ctx
    );
    Ok(())
}

#[rstest]
#[case::add(NumericOperator::Add, DecimalDType::new(11, 2))]
#[case::sub(NumericOperator::Sub, DecimalDType::new(11, 2))]
#[case::mul(NumericOperator::Mul, DecimalDType::new(21, 4))]
#[case::div(NumericOperator::Div, DecimalDType::new(16, 6))]
fn test_decimal_result_dtype(
    #[case] op: NumericOperator,
    #[case] expected: DecimalDType,
) -> VortexResult<()> {
    assert_eq!(
        result_decimal_dtype(DecimalDType::new(10, 2), op)?,
        expected
    );
    Ok(())
}

#[test]
fn test_decimal_mul_result_scale_overflow_errors() {
    assert!(result_decimal_dtype(DecimalDType::new(40, 40), NumericOperator::Mul).is_err());
}
