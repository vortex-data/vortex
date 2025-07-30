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
use vortex_error::{VortexExpect, VortexResult, VortexUnwrap, vortex_err};
use vortex_scalar::{NumericOperator, PrimitiveScalar, Scalar};

use crate::arrays::{ConstantArray, PrimitiveArray};
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
