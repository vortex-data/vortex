// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::expect_used)]

//! This example demonstrates array immutability in Vortex.
//!
//! Arrays in Vortex are immutable - you cannot modify an existing array.
//! Instead, you create new arrays with the desired changes.

use vortex::arrays::{PrimitiveArray, VarBinArray};
use vortex::buffer::buffer;
use vortex::{Array, IntoArray};

fn main() {
    // [immutability-concept]
    println!("=== Array Immutability ===\n");

    // Arrays are immutable once created
    let original = buffer![1i32, 2, 3, 4, 5].into_array();
    println!("Original array: {}", original.display_values());

    // ❌ CANNOT DO: Arrays are immutable
    // original[0] = 42;        // This doesn't exist!
    // original.set(0, 42);     // This doesn't exist either!
    // original.push(6);        // Cannot append to existing array!

    println!("\n⚠️  Arrays cannot be modified after creation!");
    println!("✅ Instead, create new arrays with the changes.\n");
    // [immutability-concept]

    // [modify-with-iter]
    // Option 1: Create modified array using iterators
    println!("=== Creating Modified Arrays with Iterators ===\n");

    // Create a new array with modifications
    let modified_values: Vec<Option<i32>> = (0..original.len())
        .map(|i| {
            if i == 0 {
                Some(42)  // Replace first element
            } else if i == 2 {
                None      // Make third element null
            } else {
                original.scalar_at(i).as_primitive().typed_value::<i32>()
            }
        })
        .chain([Some(6), Some(7)])  // Add extra elements
        .collect();

    let modified = PrimitiveArray::from_option_iter(modified_values);
    println!("Modified array: {}", modified.display_values());
    println!("Original unchanged: {}", original.display_values());
    // [modify-with-iter]

    // [modify-with-vec]
    // Option 2: Collect to Vec, modify, create new array
    println!("\n=== Creating Modified Arrays via Vec ===\n");

    // Extract values to Vec
    let mut values: Vec<Option<i32>> = (0..original.len())
        .map(|i| original.scalar_at(i).as_primitive().typed_value::<i32>())
        .collect();

    // Modify the Vec
    values[0] = Some(100);
    values.push(Some(6));
    values.remove(2);

    // Create new array from modified Vec
    let from_vec = PrimitiveArray::from_option_iter(values);
    println!("Array from modified Vec: {}", from_vec.display_values());
    println!("Original still unchanged: {}", original.display_values());
    // [modify-with-vec]

    // [string-modification]
    // String arrays are also immutable
    println!("\n=== String Array Immutability ===\n");

    let string_array = VarBinArray::from(vec!["hello", "world", "rust"]);
    println!("Original strings: {}", string_array.display_values());

    // Create modified version using iterator
    let modified_strings = vec!["hello", "vortex", "rust", "arrays"];
    let modified_array = VarBinArray::from(modified_strings);

    println!("Modified strings: {}", modified_array.display_values());
    println!("Original unchanged: {}", string_array.display_values());
    // [string-modification]

    // [functional-transformations]
    // Functional transformations create new arrays
    println!("\n=== Functional Transformations ===\n");

    let numbers = buffer![10i32, 20, 30, 40, 50].into_array();

    // Create a new array with doubled values
    let doubled_values: Vec<i32> = (0..numbers.len())
        .map(|i| {
            numbers.scalar_at(i)
                .as_primitive()
                .typed_value::<i32>()
                .unwrap_or(0) * 2
        })
        .collect();
    let doubled = PrimitiveArray::from_iter(doubled_values);

    println!("Original: {}", numbers.display_values());
    println!("Doubled: {}", doubled.display_values());

    // Filter: create new array with only values > 25
    let filtered_values: Vec<i32> = (0..numbers.len())
        .filter_map(|i| {
            numbers.scalar_at(i)
                .as_primitive()
                .typed_value::<i32>()
                .filter(|&v| v > 25)
        })
        .collect();
    let filtered = PrimitiveArray::from_iter(filtered_values);
    println!("Filtered (>25): {}", filtered.display_values());
    // [functional-transformations]

    // [slice-is-view]
    // Slicing creates a view, doesn't modify original
    println!("\n=== Slicing Creates Views ===\n");

    let data = buffer![1i32, 2, 3, 4, 5, 6, 7, 8, 9, 10].into_array();
    let slice = data.slice(2..7);

    println!("Original: {}", data.display_values());
    println!("Slice [2..7]: {}", slice.display_values());
    println!("Original unchanged: {}", data.display_values());

    // Slicing is O(1) - doesn't copy data
    println!("Original nbytes: {}", data.nbytes());
    println!("Slice nbytes: {} (shares memory)", slice.nbytes());
    // [slice-is-view]

    // [compute-returns-new]
    // Compute operations return NEW arrays, never modify existing ones
    println!("\n=== Compute Operations Return New Arrays ===\n");

    use vortex::compute::fill_null;
    use vortex::scalar::Scalar;

    // Create array with nulls using from_option_iter
    let with_nulls = PrimitiveArray::from_option_iter([
        Some(1i32),
        None,
        Some(3),
        None,
        Some(5)
    ]);
    println!("Array with nulls: {}", with_nulls.display_values());

    // fill_null returns a NEW array - original is unchanged!
    let with_nulls_array = with_nulls.into_array();
    let filled = fill_null(&with_nulls_array, &Scalar::from(99i32))
        .expect("Failed to fill nulls");

    println!("After fill_null(99):");
    println!("  Returned array: {}", filled.display_values());
    println!("  Original unchanged: {}", with_nulls_array.display_values());

    // Similarly for other operations like filter, take, etc.
    // They all return NEW arrays rather than modifying existing ones

    println!("\n📝 Note: All compute operations follow this pattern:");
    println!("   - Take immutable array as input");
    println!("   - Return NEW array as output");
    println!("   - Original array is never modified");
    // [compute-returns-new]
}