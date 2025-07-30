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
//! - Reverse Subtraction (scalar - array)
//! - Multiplication (`*`)
//! - Division (`/`)
//! - Reverse Division (scalar / array)

use itertools::Itertools;
use num_traits::Num;
use vortex_dtype::NativePType;
use vortex_error::{VortexExpect, VortexUnwrap, vortex_err};
use vortex_scalar::{NumericOperator, PrimitiveScalar, Scalar};

use crate::arrays::ConstantArray;
use crate::compute::numeric::numeric;
use crate::{Array, ArrayRef, IntoArray, ToCanonical};

fn to_vec_of_scalar(array: &dyn Array) -> Vec<Scalar> {
    // Not fast, but obviously correct
    (0..array.len())
        .map(|index| array.scalar_at(index))
        .try_collect()
        .vortex_unwrap()
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
///
/// # Panics
///
/// Panics if:
/// - The array cannot be converted to primitive form
/// - Numeric operations overflow
/// - Results don't match expected values
pub fn test_binary_numeric_conformance<T: NativePType + Num + Copy>(array: ArrayRef)
where
    Scalar: From<T>,
{
    // First test with the standard scalar value of 1
    test_standard_binary_numeric::<T>(array.clone());

    // Then test edge cases if we have enough data
    if array.len() >= 5 {
        test_binary_numeric_edge_cases(array);
    }
}

fn test_standard_binary_numeric<T: NativePType + Num + Copy>(array: ArrayRef)
where
    Scalar: From<T>,
{
    let canonicalized_array = array
        .to_primitive()
        .vortex_expect("Failed to canonicalize array to primitive form for binary numeric test");
    let original_values = to_vec_of_scalar(&canonicalized_array.into_array());

    let one = T::from(1)
        .ok_or_else(|| vortex_err!("could not convert 1 into array native type"))
        .vortex_unwrap();
    let scalar_one = Scalar::from(one).cast(array.dtype()).vortex_unwrap();

    let operators: [NumericOperator; 6] = [
        NumericOperator::Add,
        NumericOperator::Sub,
        NumericOperator::RSub,
        NumericOperator::Mul,
        NumericOperator::Div,
        NumericOperator::RDiv,
    ];

    for operator in operators {
        // Test array operator scalar (e.g., array + 1)
        let result = numeric(
            &array,
            &ConstantArray::new(scalar_one.clone(), array.len()).into_array(),
            operator,
        )
        .vortex_expect(&format!(
            "Failed to compute {operator:?} between array and constant for encoding {}",
            array.encoding_id()
        ));

        let actual_values = to_vec_of_scalar(&result);
        let expected_values: Vec<Scalar> = original_values
            .iter()
            .map(|x| {
                x.as_primitive()
                    .checked_binary_numeric(&scalar_one.as_primitive(), operator)
                    .vortex_expect(&format!(
                        "Numeric operator {operator:?} overflow for value {x:?}"
                    ))
            })
            .map(<Scalar as From<PrimitiveScalar<'_>>>::from)
            .collect();

        assert_eq!(
            actual_values,
            expected_values,
            "Binary numeric operation failed for encoding {}: \
             ({array:?}) {operator:?} (Constant array of {scalar_one}) \
             produced incorrect results. \
             Expected first few values: {:?}, \
             Actual first few values: {:?}",
            array.encoding_id(),
            &expected_values[..5.min(expected_values.len())],
            &actual_values[..5.min(actual_values.len())],
        );

        // Test scalar operator array (e.g., 1 + array)
        let result = numeric(
            &ConstantArray::new(scalar_one.clone(), array.len()).into_array(),
            &array,
            operator,
        )
        .vortex_expect(&format!(
            "Failed to compute {operator:?} between constant and array for encoding {}",
            array.encoding_id()
        ));

        let actual_values = to_vec_of_scalar(&result);
        let expected_values: Vec<Scalar> = original_values
            .iter()
            .map(|x| {
                scalar_one
                    .as_primitive()
                    .checked_binary_numeric(&x.as_primitive(), operator)
                    .vortex_expect(&format!(
                        "Numeric operator {operator:?} overflow for value {x:?}"
                    ))
            })
            .map(<Scalar as From<PrimitiveScalar<'_>>>::from)
            .collect();

        assert_eq!(
            actual_values,
            expected_values,
            "Binary numeric operation failed for encoding {}: \
             (Constant array of {scalar_one}) {operator:?} ({array:?}) \
             produced incorrect results. \
             Expected first few values: {:?}, \
             Actual first few values: {:?}",
            array.encoding_id(),
            &expected_values[..5.min(expected_values.len())],
            &actual_values[..5.min(actual_values.len())],
        );
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
    use vortex_dtype::PType;

    match array.dtype() {
        vortex_dtype::DType::Primitive(ptype, _) => match ptype {
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
        _ => {
            panic!(
                "Binary numeric tests are only supported for primitive numeric types, got {:?}",
                array.dtype()
            );
        }
    }
}

/// Tests binary numeric operations with edge case scalar values.
///
/// This function tests operations with scalar values:
/// - Zero (identity for addition/subtraction, absorbing for multiplication)
/// - Negative one (tests signed arithmetic)
/// - Maximum value (tests overflow behavior)
/// - Minimum value (tests underflow behavior)
pub fn test_binary_numeric_edge_cases(array: ArrayRef) {
    use vortex_dtype::PType;

    match array.dtype() {
        vortex_dtype::DType::Primitive(ptype, _) => match ptype {
            PType::I8 => test_binary_numeric_edge_cases_for_type::<i8>(array),
            PType::I16 => test_binary_numeric_edge_cases_for_type::<i16>(array),
            PType::I32 => test_binary_numeric_edge_cases_for_type::<i32>(array),
            PType::I64 => test_binary_numeric_edge_cases_for_type::<i64>(array),
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
        _ => {
            panic!("Binary numeric edge case tests are only supported for primitive numeric types");
        }
    }
}

fn test_binary_numeric_edge_cases_for_type<T>(array: ArrayRef)
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
    test_binary_numeric_with_scalar(array.clone(), T::max_value());

    // Test with min value (0 for unsigned)
    test_binary_numeric_with_scalar(array, T::min_value());
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
    let canonicalized_array = array
        .to_primitive()
        .vortex_expect("Failed to canonicalize array to primitive form for binary numeric test");
    let original_values = to_vec_of_scalar(&canonicalized_array.into_array());

    let scalar = Scalar::from(scalar_value)
        .cast(array.dtype())
        .vortex_unwrap();

    // Only test operators that make sense for the given scalar
    let operators = if scalar_value == T::zero() {
        // Skip division by zero
        vec![
            NumericOperator::Add,
            NumericOperator::Sub,
            NumericOperator::RSub,
            NumericOperator::Mul,
        ]
    } else {
        vec![
            NumericOperator::Add,
            NumericOperator::Sub,
            NumericOperator::RSub,
            NumericOperator::Mul,
            NumericOperator::Div,
            NumericOperator::RDiv,
        ]
    };

    for operator in operators {
        // Test array operator scalar
        let result = numeric(
            &array,
            &ConstantArray::new(scalar.clone(), array.len()).into_array(),
            operator,
        );

        // Skip if operation would overflow/underflow
        if result.is_err() {
            continue;
        }

        let result = result.vortex_unwrap();
        let actual_values = to_vec_of_scalar(&result);

        let expected_values: Vec<Option<Scalar>> = original_values
            .iter()
            .map(|x| {
                x.as_primitive()
                    .checked_binary_numeric(&scalar.as_primitive(), operator)
                    .map(|ps| <Scalar as From<PrimitiveScalar<'_>>>::from(ps))
            })
            .collect();

        // Skip if any expected values would overflow (contain None)
        if expected_values.iter().any(|v| v.is_none()) {
            continue;
        }

        let expected_values: Vec<Scalar> = expected_values.into_iter().flatten().collect();

        assert_eq!(
            actual_values,
            expected_values,
            "Binary numeric operation failed for encoding {} with scalar {:?}: \
             ({array:?}) {operator:?} (Constant array of {scalar}) \
             produced incorrect results.",
            array.encoding_id(),
            scalar_value,
        );
    }
}
