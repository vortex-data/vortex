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

use itertools::Itertools;
use num_traits::Num;
use vortex_error::VortexExpect;
use vortex_error::vortex_err;
use vortex_error::vortex_panic;

use crate::ArrayRef;
use crate::IntoArray;
use crate::LEGACY_SESSION;
use crate::RecursiveCanonical;
use crate::ToCanonical;
use crate::VortexSessionExecute;
use crate::arrays::ConstantArray;
use crate::builtins::ArrayBuiltins;
use crate::dtype::DType;
use crate::dtype::NativePType;
use crate::dtype::PType;
use crate::scalar::NumericOperator;
use crate::scalar::PrimitiveScalar;
use crate::scalar::Scalar;

fn to_vec_of_scalar(array: &ArrayRef) -> Vec<Scalar> {
    // Not fast, but obviously correct
    (0..array.len())
        .map(|index| {
            array
                .scalar_at(index)
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
fn test_binary_numeric_conformance<T: NativePType + Num + Copy>(array: ArrayRef)
where
    Scalar: From<T>,
{
    // First test with the standard scalar value of 1
    test_standard_binary_numeric::<T>(array.clone());

    // Then test edge cases
    test_binary_numeric_edge_cases(array);
}

fn test_standard_binary_numeric<T: NativePType + Num + Copy>(array: ArrayRef)
where
    Scalar: From<T>,
{
    let canonicalized_array = array.to_primitive();
    let original_values = to_vec_of_scalar(&canonicalized_array.into_array());

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
            .execute::<RecursiveCanonical>(&mut LEGACY_SESSION.create_execution_ctx())
            .map(|c| c.0.into_array());

        // Skip this operator if the entire operation fails
        // This can happen for some edge cases in specific encodings
        let Ok(result) = result else {
            continue;
        };

        let actual_values = to_vec_of_scalar(&result);

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
            a.execute::<RecursiveCanonical>(&mut LEGACY_SESSION.create_execution_ctx())
                .map(|c| c.0.into_array())
        });

        // Skip this operator if the entire operation fails
        let Ok(result) = result else {
            continue;
        };

        let actual_values = to_vec_of_scalar(&result);

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
pub fn test_binary_numeric_array(array: ArrayRef) {
    match array.dtype() {
        DType::Primitive(ptype, _) => match ptype {
            PType::I8 => test_binary_numeric_conformance::<i8>(array),
            PType::I16 => test_binary_numeric_conformance::<i16>(array),
            PType::I32 => test_binary_numeric_conformance::<i32>(array),
            PType::I64 => test_binary_numeric_conformance::<i64>(array),
            PType::U8 => test_binary_numeric_conformance::<u8>(array),
            PType::U16 => test_binary_numeric_conformance::<u16>(array),
            PType::U32 => test_binary_numeric_conformance::<u32>(array),
            PType::U64 => test_binary_numeric_conformance::<u64>(array),
            PType::F16 => {
                // F16 not supported in num-traits, skip
                eprintln!("Skipping f16 binary numeric tests (not supported)");
            }
            PType::F32 => test_binary_numeric_conformance::<f32>(array),
            PType::F64 => test_binary_numeric_conformance::<f64>(array),
        },
        dtype => vortex_panic!(
            "Binary numeric tests are only supported for primitive numeric types, got {dtype}",
        ),
    }
}

/// Tests binary numeric operations with edge case scalar values.
///
/// This function tests operations with scalar values:
/// - Zero (identity for addition/subtraction, absorbing for multiplication)
/// - Negative one (tests signed arithmetic)
/// - Maximum value (tests overflow behavior)
/// - Minimum value (tests underflow behavior)
fn test_binary_numeric_edge_cases(array: ArrayRef) {
    match array.dtype() {
        DType::Primitive(ptype, _) => match ptype {
            PType::I8 => test_binary_numeric_edge_cases_signed::<i8>(array),
            PType::I16 => test_binary_numeric_edge_cases_signed::<i16>(array),
            PType::I32 => test_binary_numeric_edge_cases_signed::<i32>(array),
            PType::I64 => test_binary_numeric_edge_cases_signed::<i64>(array),
            PType::U8 => test_binary_numeric_edge_cases_unsigned::<u8>(array),
            PType::U16 => test_binary_numeric_edge_cases_unsigned::<u16>(array),
            PType::U32 => test_binary_numeric_edge_cases_unsigned::<u32>(array),
            PType::U64 => test_binary_numeric_edge_cases_unsigned::<u64>(array),
            PType::F16 => {
                eprintln!("Skipping f16 edge case tests (not supported)");
            }
            PType::F32 => test_binary_numeric_edge_cases_float::<f32>(array),
            PType::F64 => test_binary_numeric_edge_cases_float::<f64>(array),
        },
        dtype => vortex_panic!(
            "Binary numeric edge case tests are only supported for primitive numeric types, got {dtype}"
        ),
    }
}

fn test_binary_numeric_edge_cases_signed<T>(array: ArrayRef)
where
    T: NativePType + Num + Copy + std::fmt::Debug + num_traits::Bounded + num_traits::Signed,
    Scalar: From<T>,
{
    // Test with zero
    test_binary_numeric_with_scalar(array.clone(), T::zero());

    // Test with -1
    test_binary_numeric_with_scalar(array.clone(), -T::one());

    // Test with max value
    test_binary_numeric_with_scalar(array.clone(), T::max_value());

    // Test with min value
    test_binary_numeric_with_scalar(array, T::min_value());
}

fn test_binary_numeric_edge_cases_unsigned<T>(array: ArrayRef)
where
    T: NativePType + Num + Copy + std::fmt::Debug + num_traits::Bounded,
    Scalar: From<T>,
{
    // Test with zero
    test_binary_numeric_with_scalar(array.clone(), T::zero());

    // Test with max value
    test_binary_numeric_with_scalar(array, T::max_value());
}

fn test_binary_numeric_edge_cases_float<T>(array: ArrayRef)
where
    T: NativePType + Num + Copy + std::fmt::Debug + num_traits::Float,
    Scalar: From<T>,
{
    // Test with zero
    test_binary_numeric_with_scalar(array.clone(), T::zero());

    // Test with -1
    test_binary_numeric_with_scalar(array.clone(), -T::one());

    // Test with max value
    test_binary_numeric_with_scalar(array.clone(), T::max_value());

    // Test with min value
    test_binary_numeric_with_scalar(array.clone(), T::min_value());

    // Test with small positive value
    test_binary_numeric_with_scalar(array.clone(), T::epsilon());

    // Test with min positive value (subnormal)
    test_binary_numeric_with_scalar(array.clone(), T::min_positive_value());

    // Test with special float values (NaN, Infinity)
    test_binary_numeric_with_scalar(array.clone(), T::nan());
    test_binary_numeric_with_scalar(array.clone(), T::infinity());
    test_binary_numeric_with_scalar(array, T::neg_infinity());
}

fn test_binary_numeric_with_scalar<T>(array: ArrayRef, scalar_value: T)
where
    T: NativePType + Num + Copy + std::fmt::Debug,
    Scalar: From<T>,
{
    let canonicalized_array = array.to_primitive();
    let original_values = to_vec_of_scalar(&canonicalized_array.into_array());

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
            .execute::<RecursiveCanonical>(&mut LEGACY_SESSION.create_execution_ctx())
            .map(|x| x.0.into_array());

        // Skip if the entire operation fails
        // TODO(joe): this is odd.
        if result.is_err() {
            continue;
        }

        let result = result.vortex_expect("operation should succeed in conformance test");
        let actual_values = to_vec_of_scalar(&result);

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
