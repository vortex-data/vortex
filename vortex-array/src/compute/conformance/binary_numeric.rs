// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! # Binary Numeric Conformance Tests
//!
//! This module provides conformance testing for binary numeric operations on Vortex arrays.
//! It ensures that all numeric array encodings produce identical results when performing
//! arithmetic operations (add, subtract, multiply, divide).
//!
//! ## Test Strategy
//!
//! For each array encoding, we test:
//! 1. All binary numeric operators against a constant scalar value
//! 2. Both left-hand and right-hand side operations (e.g., array + 1 and 1 + array)
//! 3. That results match the canonical primitive array implementation
//!
//! ## Supported Operations
//!
//! - Addition (`+`)
//! - Subtraction (`-`)
//! - Multiplication (`*`)
//! - Division (`/`)

use std::fmt::Debug;

use itertools::Itertools;
use num_traits::Bounded;
use num_traits::CheckedAdd;
use num_traits::CheckedDiv;
use num_traits::CheckedMul;
use num_traits::CheckedSub;
use num_traits::Float;
use num_traits::Num;
use num_traits::Signed;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::Canonical;
use crate::ExecutionCtx;
use crate::IntoArray;
use crate::RecursiveCanonical;
use crate::arrays::ConstantArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::DecimalDType;
use crate::dtype::NativeDecimalType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::dtype::i256;
use crate::scalar::DecimalValue;
use crate::scalar::NumericOperator;
use crate::scalar::PrimitiveScalar;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::numeric_op_result_decimal_dtype;

fn to_vec_of_scalar(array: &ArrayRef, ctx: &mut ExecutionCtx) -> Vec<Scalar> {
    // Not fast, but obviously correct
    (0..array.len())
        .map(|index| {
            array
                .execute_scalar(index, ctx)
                .vortex_expect("scalar_at should succeed in conformance test")
        })
        .collect_vec()
}

/// Tests binary numeric operations for conformance across array encodings.
///
/// # Type Parameters
///
/// * `T` - The native numeric type (e.g., i32, f64) that the array contains
///
/// # Arguments
///
/// * `array` - The array to test, which should contain numeric values of type `T`
///
/// # Test Details
///
/// This function:
/// 1. Canonicalizes the input array to primitive form to get expected values
/// 2. Tests all binary numeric operators against a constant value of 1
/// 3. Verifies results match the expected primitive array computation
/// 4. Tests both array-operator-scalar and scalar-operator-array forms
/// 5. Gracefully skips operations that would cause overflow/underflow
///
/// # Panics
///
/// Panics if:
/// - The array cannot be converted to primitive form
/// - Results don't match expected values (for operations that don't overflow)
fn test_binary_numeric_conformance<T: NativePType + Num + Copy>(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) where
    Scalar: From<T>,
{
    // First test with the standard scalar value of 1
    test_standard_binary_numeric::<T>(array, ctx);

    // Then test edge cases
    test_binary_numeric_edge_cases(array, ctx);
}

fn test_standard_binary_numeric<T: NativePType + Num + Copy>(
    array: &ArrayRef,
    ctx: &mut ExecutionCtx,
) where
    Scalar: From<T>,
{
    let canonicalized_array = array
        .clone()
        .execute::<Canonical>(ctx)
        .vortex_expect("Must be able to canonicalise")
        .into_array();
    let original_values = to_vec_of_scalar(&canonicalized_array, ctx);

    let one = T::from(1)
        .ok_or_else(|| vortex_err!("could not convert 1 into array native type"))
        .vortex_expect("operation should succeed in conformance test");
    let scalar_one = Scalar::from(one)
        .cast(array.dtype())
        .vortex_expect("operation should succeed in conformance test");

    let operators: [NumericOperator; 4] = [
        NumericOperator::Add,
        NumericOperator::Sub,
        NumericOperator::Mul,
        NumericOperator::Div,
    ];

    for operator in operators {
        let op = operator;
        let rhs_const = ConstantArray::new(scalar_one.clone(), array.len()).into_array();

        // Test array operator scalar (e.g., array + 1)
        let result = array
            .binary(rhs_const.clone(), op.into())
            .vortex_expect("apply shouldn't fail")
            .execute::<RecursiveCanonical>(ctx)
            .map(|c| c.0.into_array());

        // Skip this operator if the entire operation fails
        // This can happen for some edge cases in specific encodings
        let Ok(result) = result else {
            continue;
        };

        let actual_values = to_vec_of_scalar(&result, ctx);

        // Check each element for overflow/underflow
        let expected_results: Vec<Option<Scalar>> = original_values
            .iter()
            .map(|x| {
                x.as_primitive()
                    .checked_binary_numeric(&scalar_one.as_primitive(), op)
                    .map(<Scalar as From<PrimitiveScalar<'_>>>::from)
            })
            .collect();

        // For elements that didn't overflow, check they match
        for (idx, (actual, expected)) in actual_values.iter().zip(&expected_results).enumerate() {
            if let Some(expected_value) = expected {
                assert_eq!(
                    actual,
                    expected_value,
                    "Binary numeric operation failed for encoding {} at index {}: \
                     ({array:?})[{idx}] {operator:?} {scalar_one} \
                     expected {expected_value:?}, got {actual:?}",
                    array.encoding_id(),
                    idx,
                );
            }
        }

        // Test scalar operator array (e.g., 1 + array)
        let result = rhs_const.binary(array.clone(), op.into()).and_then(|a| {
            a.execute::<RecursiveCanonical>(ctx)
                .map(|c| c.0.into_array())
        });

        // Skip this operator if the entire operation fails
        let Ok(result) = result else {
            continue;
        };

        let actual_values = to_vec_of_scalar(&result, ctx);

        // Check each element for overflow/underflow
        let expected_results: Vec<Option<Scalar>> = original_values
            .iter()
            .map(|x| {
                scalar_one
                    .as_primitive()
                    .checked_binary_numeric(&x.as_primitive(), op)
                    .map(<Scalar as From<PrimitiveScalar<'_>>>::from)
            })
            .collect();

        // For elements that didn't overflow, check they match
        for (idx, (actual, expected)) in actual_values.iter().zip(&expected_results).enumerate() {
            if let Some(expected_value) = expected {
                assert_eq!(
                    actual,
                    expected_value,
                    "Binary numeric operation failed for encoding {} at index {}: \
                     {scalar_one} {operator:?} ({array:?})[{idx}] \
                     expected {expected_value:?}, got {actual:?}",
                    array.encoding_id(),
                    idx,
                );
            }
        }
    }
}

/// Entry point for binary numeric conformance testing for any array type.
///
/// This function automatically detects the array's numeric type and runs
/// the appropriate tests. It's designed to be called from rstest parameterized
/// tests without requiring explicit type parameters.
///
/// # Example
///
/// ```ignore
/// #[rstest]
/// #[case::i32_array(create_i32_array())]
/// #[case::f64_array(create_f64_array())]
/// fn test_my_encoding_binary_numeric(#[case] array: MyArray) {
///     test_binary_numeric_array(array.into_array());
/// }
/// ```
pub fn test_binary_numeric_array(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    match array.dtype() {
        DType::Primitive(ptype, _) => match ptype {
            PType::I8 => test_binary_numeric_conformance::<i8>(array, ctx),
            PType::I16 => test_binary_numeric_conformance::<i16>(array, ctx),
            PType::I32 => test_binary_numeric_conformance::<i32>(array, ctx),
            PType::I64 => test_binary_numeric_conformance::<i64>(array, ctx),
            PType::U8 => test_binary_numeric_conformance::<u8>(array, ctx),
            PType::U16 => test_binary_numeric_conformance::<u16>(array, ctx),
            PType::U32 => test_binary_numeric_conformance::<u32>(array, ctx),
            PType::U64 => test_binary_numeric_conformance::<u64>(array, ctx),
            PType::F16 => {
                // F16 not supported in num-traits, skip
                eprintln!("Skipping f16 binary numeric tests (not supported)");
            }
            PType::F32 => test_binary_numeric_conformance::<f32>(array, ctx),
            PType::F64 => test_binary_numeric_conformance::<f64>(array, ctx),
        },
        DType::Decimal(decimal_dtype, _) => {
            test_binary_numeric_conformance_decimal(array, *decimal_dtype, ctx)
        }
        dtype => vortex_panic!(
            "Binary numeric tests are only supported for primitive and decimal types, got {dtype}",
        ),
    }
}

/// Tests binary numeric operations on a decimal array against the decimal scalar
/// implementation, using a set of representative constants: stored `0`, `1`, `-1`, the decimal
/// `1.0` (when it fits the precision), and the largest stored value for the precision.
fn test_binary_numeric_conformance_decimal(
    array: &ArrayRef,
    decimal_dtype: DecimalDType,
    ctx: &mut ExecutionCtx,
) {
    let precision = decimal_dtype.precision() as usize;
    let prec_max = <i256 as NativeDecimalType>::MAX_BY_PRECISION[precision];

    let mut constants = vec![
        i256::from_i128(0),
        i256::from_i128(1),
        i256::from_i128(-1),
        prec_max,
    ];
    // The decimal value 1.0 (stored 10^s), when representable within the precision.
    if decimal_dtype.scale() >= 0
        && let Some(one) = i256::from_i128(10).checked_pow(decimal_dtype.scale() as u32)
        && one <= prec_max
    {
        constants.push(one);
    }

    for constant in constants {
        let value = DecimalValue::try_from_i256(constant, decimal_dtype)
            .vortex_expect("conformance constants fit the precision");
        test_decimal_binary_numeric_with_scalar(array, value, decimal_dtype, ctx);
    }
}

fn test_decimal_binary_numeric_with_scalar(
    array: &ArrayRef,
    value: DecimalValue,
    decimal_dtype: DecimalDType,
    ctx: &mut ExecutionCtx,
) {
    let canonicalized_array = array
        .clone()
        .execute::<Canonical>(ctx)
        .vortex_expect("Must be able to canonicalise")
        .into_array();
    let original_values = to_vec_of_scalar(&canonicalized_array, ctx);

    let scalar = Scalar::decimal(value, decimal_dtype, array.dtype().nullability());

    let mut operators = vec![
        NumericOperator::Add,
        NumericOperator::Sub,
        NumericOperator::Mul,
    ];
    if !value.is_zero() {
        operators.push(NumericOperator::Div);
    }

    for operator in operators {
        for lhs_is_array in [true, false] {
            test_decimal_binary_numeric_direction(
                array,
                &original_values,
                &scalar,
                decimal_dtype,
                operator,
                lhs_is_array,
                ctx,
            );
        }
    }
}

fn test_decimal_binary_numeric_direction(
    array: &ArrayRef,
    original_values: &[Scalar],
    scalar: &Scalar,
    decimal_dtype: DecimalDType,
    operator: NumericOperator,
    lhs_is_array: bool,
    ctx: &mut ExecutionCtx,
) {
    let result_decimal_dtype = numeric_op_result_decimal_dtype(decimal_dtype, operator)
        .vortex_expect("decimal arithmetic must have a result dtype");
    let result_dtype = DType::Decimal(result_decimal_dtype, array.dtype().nullability());
    let expected_results = expected_decimal_results(
        original_values,
        scalar,
        operator,
        result_decimal_dtype,
        &result_dtype,
        lhs_is_array,
    );

    let constant = ConstantArray::new(scalar.clone(), array.len()).into_array();
    let (lhs, rhs) = if lhs_is_array {
        (array.clone(), constant)
    } else {
        (constant, array.clone())
    };
    let result = lhs
        .binary(rhs, operator.into())
        .vortex_expect("binary decimal op shouldn't fail")
        .execute::<RecursiveCanonical>(ctx)
        .map(|c| c.0.into_array());

    assert_decimal_results(
        result,
        &expected_results,
        array,
        scalar,
        operator,
        lhs_is_array,
        ctx,
    );
}

fn expected_decimal_results(
    original_values: &[Scalar],
    scalar: &Scalar,
    operator: NumericOperator,
    result_decimal_dtype: DecimalDType,
    result_dtype: &DType,
    lhs_is_array: bool,
) -> Vec<Option<Scalar>> {
    original_values
        .iter()
        .map(|value| {
            let (lhs, rhs) = if lhs_is_array {
                (value.as_decimal(), scalar.as_decimal())
            } else {
                (scalar.as_decimal(), value.as_decimal())
            };
            let (Some(lhs), Some(rhs)) = (lhs.decimal_value(), rhs.decimal_value()) else {
                return Some(Scalar::null(result_dtype.clone()));
            };
            let lhs = lhs.as_i256();
            let rhs = rhs.as_i256();
            let value = match operator {
                NumericOperator::Add => lhs.checked_add(&rhs),
                NumericOperator::Sub => lhs.checked_sub(&rhs),
                NumericOperator::Mul => lhs.checked_mul(&rhs),
                NumericOperator::Div => {
                    let scale_power = result_decimal_dtype.scale();
                    let factor =
                        i256::from_i128(10).checked_pow(scale_power.unsigned_abs() as u32)?;
                    if scale_power >= 0 {
                        lhs.checked_mul(&factor)?.checked_div(&rhs)
                    } else {
                        lhs.checked_div(&rhs.checked_mul(&factor)?)
                    }
                }
            }?;
            let value = DecimalValue::try_from_i256(value, result_decimal_dtype).ok()?;
            Some(Scalar::decimal(
                value,
                result_decimal_dtype,
                result_dtype.nullability(),
            ))
        })
        .collect()
}

fn assert_decimal_results(
    result: vortex_error::VortexResult<ArrayRef>,
    expected_results: &[Option<Scalar>],
    array: &ArrayRef,
    scalar: &Scalar,
    operator: NumericOperator,
    lhs_is_array: bool,
    ctx: &mut ExecutionCtx,
) {
    if expected_results.iter().any(Option::is_none) {
        assert!(
            result.is_err(),
            "Decimal binary numeric operation should overflow for encoding {}: \
             {operator:?} {scalar} (lhs_is_array: {lhs_is_array})",
            array.encoding_id(),
        );
        return;
    }

    let result = result.unwrap_or_else(|err| {
        vortex_panic!(
            "Decimal binary numeric operation unexpectedly failed for encoding {}: \
             {operator:?} {scalar} (lhs_is_array: {lhs_is_array}): {err}",
            array.encoding_id(),
        )
    });

    let actual_values = to_vec_of_scalar(&result, ctx);
    for (idx, (actual, expected)) in actual_values.iter().zip(expected_results).enumerate() {
        let expected_value = expected
            .as_ref()
            .vortex_expect("non-overflowing decimal lane must have an expected value");
        assert_eq!(
            actual,
            expected_value,
            "Decimal binary numeric operation failed for encoding {} at index {}: \
             ({array:?})[{idx}] {operator:?} {scalar} (lhs_is_array: {lhs_is_array}) \
             expected {expected_value:?}, got {actual:?}",
            array.encoding_id(),
            idx,
        );
    }
}

/// Tests binary numeric operations with edge case scalar values.
///
/// This function tests operations with scalar values:
/// - Zero (identity for addition/subtraction, absorbing for multiplication)
/// - Negative one (tests signed arithmetic)
/// - Maximum value (tests overflow behavior)
/// - Minimum value (tests underflow behavior)
fn test_binary_numeric_edge_cases(array: &ArrayRef, ctx: &mut ExecutionCtx) {
    match array.dtype() {
        DType::Primitive(ptype, _) => match ptype {
            PType::I8 => test_binary_numeric_edge_cases_signed::<i8>(array, ctx),
            PType::I16 => test_binary_numeric_edge_cases_signed::<i16>(array, ctx),
            PType::I32 => test_binary_numeric_edge_cases_signed::<i32>(array, ctx),
            PType::I64 => test_binary_numeric_edge_cases_signed::<i64>(array, ctx),
            PType::U8 => test_binary_numeric_edge_cases_unsigned::<u8>(array, ctx),
            PType::U16 => test_binary_numeric_edge_cases_unsigned::<u16>(array, ctx),
            PType::U32 => test_binary_numeric_edge_cases_unsigned::<u32>(array, ctx),
            PType::U64 => test_binary_numeric_edge_cases_unsigned::<u64>(array, ctx),
            PType::F16 => {
                eprintln!("Skipping f16 edge case tests (not supported)");
            }
            PType::F32 => test_binary_numeric_edge_cases_float::<f32>(array, ctx),
            PType::F64 => test_binary_numeric_edge_cases_float::<f64>(array, ctx),
        },
        dtype => vortex_panic!(
            "Binary numeric edge case tests are only supported for primitive numeric types, got {dtype}"
        ),
    }
}

fn test_binary_numeric_edge_cases_signed<T>(array: &ArrayRef, ctx: &mut ExecutionCtx)
where
    T: NativePType + Num + Copy + Debug + Bounded + Signed,
    Scalar: From<T>,
{
    // Test with zero
    test_binary_numeric_with_scalar(array, T::zero(), ctx);

    // Test with -1
    test_binary_numeric_with_scalar(array, -T::one(), ctx);

    // Test with max value
    test_binary_numeric_with_scalar(array, T::max_value(), ctx);

    // Test with min value
    test_binary_numeric_with_scalar(array, T::min_value(), ctx);
}

fn test_binary_numeric_edge_cases_unsigned<T>(array: &ArrayRef, ctx: &mut ExecutionCtx)
where
    T: NativePType + Num + Copy + Debug + Bounded,
    Scalar: From<T>,
{
    // Test with zero
    test_binary_numeric_with_scalar(array, T::zero(), ctx);

    // Test with max value
    test_binary_numeric_with_scalar(array, T::max_value(), ctx);
}

fn test_binary_numeric_edge_cases_float<T>(array: &ArrayRef, ctx: &mut ExecutionCtx)
where
    T: NativePType + Num + Copy + Debug + Float,
    Scalar: From<T>,
{
    // Test with zero
    test_binary_numeric_with_scalar(array, T::zero(), ctx);

    // Test with -1
    test_binary_numeric_with_scalar(array, -T::one(), ctx);

    // Test with max value
    test_binary_numeric_with_scalar(array, T::max_value(), ctx);

    // Test with min value
    test_binary_numeric_with_scalar(array, T::min_value(), ctx);

    // Test with small positive value
    test_binary_numeric_with_scalar(array, T::epsilon(), ctx);

    // Test with min positive value (subnormal)
    test_binary_numeric_with_scalar(array, T::min_positive_value(), ctx);

    // Test with special float values (NaN, Infinity)
    test_binary_numeric_with_scalar(array, T::nan(), ctx);
    test_binary_numeric_with_scalar(array, T::infinity(), ctx);
    test_binary_numeric_with_scalar(array, T::neg_infinity(), ctx);
}

fn test_binary_numeric_with_scalar<T>(array: &ArrayRef, scalar_value: T, ctx: &mut ExecutionCtx)
where
    T: NativePType + Num + Copy + Debug,
    Scalar: From<T>,
{
    let canonicalized_array = array
        .clone()
        .execute::<Canonical>(ctx)
        .vortex_expect("Must be able to canonicalise")
        .into_array();
    let original_values = to_vec_of_scalar(&canonicalized_array, ctx);

    let scalar = Scalar::from(scalar_value)
        .cast(array.dtype())
        .vortex_expect("operation should succeed in conformance test");

    // Only test operators that make sense for the given scalar
    let operators = if scalar_value == T::zero() {
        // Skip division by zero
        vec![
            NumericOperator::Add,
            NumericOperator::Sub,
            NumericOperator::Mul,
        ]
    } else {
        vec![
            NumericOperator::Add,
            NumericOperator::Sub,
            NumericOperator::Mul,
            NumericOperator::Div,
        ]
    };

    for operator in operators {
        let op = operator;
        let rhs_const = ConstantArray::new(scalar.clone(), array.len()).into_array();

        // Test array operator scalar
        let result = array
            .binary(rhs_const, op.into())
            .vortex_expect("apply failed")
            .execute::<RecursiveCanonical>(ctx)
            .map(|x| x.0.into_array());

        // Skip if the entire operation fails
        // TODO(joe): this is odd.
        if result.is_err() {
            continue;
        }

        let result = result.vortex_expect("operation should succeed in conformance test");
        let actual_values = to_vec_of_scalar(&result, ctx);

        // Check each element for overflow/underflow
        let expected_results: Vec<Option<Scalar>> = original_values
            .iter()
            .map(|x| {
                x.as_primitive()
                    .checked_binary_numeric(&scalar.as_primitive(), op)
                    .map(<Scalar as From<PrimitiveScalar<'_>>>::from)
            })
            .collect();

        // For elements that didn't overflow, check they match
        for (idx, (actual, expected)) in actual_values.iter().zip(&expected_results).enumerate() {
            if let Some(expected_value) = expected {
                assert_eq!(
                    actual,
                    expected_value,
                    "Binary numeric operation failed for encoding {} at index {} with scalar {:?}: \
                     ({array:?})[{idx}] {operator:?} {scalar} \
                     expected {expected_value:?}, got {actual:?}",
                    array.encoding_id(),
                    idx,
                    scalar_value,
                );
            }
        }
    }
}
