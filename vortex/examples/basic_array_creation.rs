// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors


#![allow(clippy::expect_used)]
//! This example demonstrates how to create basic Vortex arrays.
//!
//! Vortex supports many array types. This example covers the most common cases:
//! - Primitive arrays (integers, floats)
//! - Boolean arrays
//! - Null arrays
//! - Arrays with and without null values

use vortex::arrays::{BoolArray, NullArray, PrimitiveArray};
use vortex::buffer::buffer;
use vortex::validity::Validity;
use vortex::{Array, IntoArray};

fn main() {
    // [primitive-int]
    // Create a primitive integer array using the buffer! macro
    let int_array = buffer![1i32, 2, 3, 4, 5].into_array();
    println!("Integer array: {}", int_array.display_values());
    // Output: [1i32, 2i32, 3i32, 4i32, 5i32]
    // [primitive-int]

    // [primitive-float]
    // Create a primitive float array
    let float_array = buffer![1.0f64, 2.5, 3.7, 4.0, 5.5].into_array();
    println!("Float array: {}", float_array.display_values());
    // [primitive-float]

    // [primitive-unsigned]
    // Create unsigned integer arrays
    let uint_array = buffer![10u64, 20, 30, 40, 50].into_array();
    println!("Unsigned array: {}", uint_array.display_values());
    // [primitive-unsigned]

    // [primitive-with-validity]
    // Create an array with explicit validity (nullable values)
    // First, create the buffer of values
    let values = buffer![1i32, 2, 3, 4, 5];

    // Then specify which values are valid using a boolean mask
    let validity: Validity = [true, false, true, true, false].into_iter().collect();

    let nullable_array = PrimitiveArray::new(values, validity);
    println!("Nullable array: {}", nullable_array.display_values());
    // Output shows null where validity is false
    // [primitive-with-validity]

    // [primitive-nonnullable]
    // Create an array that explicitly has no nulls
    let non_null_array = PrimitiveArray::new(buffer![1i32, 2, 3], Validity::NonNullable);
    println!("Non-nullable array: {}", non_null_array.display_values());
    // [primitive-nonnullable]

    // [bool-array]
    // Create a boolean array
    let bool_array: BoolArray = [true, false, true, true, false].into_iter().collect();
    println!("Boolean array: {}", bool_array.display_values());
    // [bool-array]

    // [bool-with-validity]
    // Boolean arrays with null values require more complex construction
    // See struct arrays example for patterns with validity
    // [bool-with-validity]

    // [null-array]
    // Create an array of all nulls
    let null_array = NullArray::new(5);
    println!("Null array (length 5): {}", null_array.display_values());
    // [null-array]

    // [array-properties]
    // All arrays have common properties
    println!("\nArray properties:");
    println!("  Length: {}", int_array.len());
    println!("  DType: {}", int_array.dtype());
    println!("  Encoding: {}", int_array.encoding().id());
    println!("  Nbytes: {}", int_array.nbytes());
    // [array-properties]

    // [constant-array]
    // You can also create constant arrays (all same value) efficiently
    let constant = buffer![42u64; 1000].into_array();
    println!(
        "\nConstant array of 1000 42s, nbytes: {}",
        constant.nbytes()
    );
    // [constant-array]
}
