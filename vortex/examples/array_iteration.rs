// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::expect_used)]
#![allow(clippy::use_debug)]
#![allow(clippy::if_then_some_else_none)]

//! This example demonstrates how to iterate over arrays and access individual elements.
//!
//! Vortex provides several ways to access array data:
//! - scalar_at(index): Get a scalar value at a specific index
//! - Iterator pattern: Iterate over array chunks
//! - Specialized accessors for specific array types

use vortex::arrays::{PrimitiveArray, VarBinArray};
use vortex::buffer::buffer;
use vortex::validity::Validity;
use vortex::{Array, IntoArray};

fn main() {
    // [scalar-at]
    // Access individual elements using scalar_at
    let array = buffer![10i32, 20, 30, 40, 50].into_array();

    println!("=== Accessing Individual Elements ===");
    for i in 0..array.len() {
        let scalar = array.scalar_at(i);
        println!("Element {}: {}", i, scalar);
    }
    // [scalar-at]

    // [typed-values]
    // Extract typed values from scalars
    println!("\n=== Extracting Typed Values ===");

    let int_array = buffer![1i32, 2, 3, 4, 5].into_array();
    for i in 0..int_array.len() {
        let scalar = int_array.scalar_at(i);
        // Use as_primitive() to get a PrimitiveScalar view
        if let Some(value) = scalar.as_primitive().typed_value::<i32>() {
            println!("Value {}: {}", i, value);
        }
    }
    // [typed-values]

    // [iterate-with-validity]
    // Handle nullable values during iteration
    println!("\n=== Iterating with Nulls ===");

    let validity: Validity = [true, false, true, false, true].into_iter().collect();
    let nullable_array = PrimitiveArray::new(buffer![1i32, 2, 3, 4, 5], validity);

    for i in 0..nullable_array.len() {
        if nullable_array.is_valid(i) {
            if let Some(value) = nullable_array
                .scalar_at(i)
                .as_primitive()
                .typed_value::<i32>()
            {
                println!("Index {}: {}", i, value);
            }
        } else {
            println!("Index {}: null", i);
        }
    }
    // [iterate-with-validity]

    // [slice-array]
    // Slicing creates a new array view without copying
    println!("\n=== Slicing Arrays ===");

    let original = buffer![0i32, 1, 2, 3, 4, 5, 6, 7, 8, 9].into_array();
    println!("Original: {}", original.display_values());

    // Slice from index 2 to 7 (exclusive)
    let sliced = original.slice(2..7);
    println!("Sliced [2..7]: {}", sliced.display_values());

    // Slicing is O(1) - it doesn't copy data
    println!("Slice nbytes: {}", sliced.nbytes());
    // [slice-array]

    // [iterate-strings]
    // Iterate over string arrays using scalar_at
    println!("\n=== Iterating String Arrays ===");

    let strings = VarBinArray::from(vec!["hello", "world", "vortex"]).into_array();

    for i in 0..strings.len() {
        let scalar = strings.scalar_at(i);
        if let Some(string_value) = scalar.as_utf8().value() {
            println!("String {}: {}", i, string_value.as_str());
        }
    }
    // [iterate-strings]

    // [array-accessor]
    // Use ArrayAccessor for efficient iteration over VarBinArray
    println!("\n=== ArrayAccessor Pattern ===");

    use vortex::accessor::ArrayAccessor;

    let varbin = VarBinArray::from(vec!["apple", "banana", "cherry"]);

    // Convert bytes to UTF-8 strings using with_iterator
    let collected = varbin.with_iterator(|iter| {
        iter.map(|bytes_opt| {
            bytes_opt.map(|bytes| {
                String::from_utf8(bytes.to_vec()).expect("Invalid UTF-8")
            })
        })
        .collect::<Vec<_>>()
    }).expect("Failed to iterate");

    println!("Collected strings: {:?}", collected);

    // With nulls - use flatten to skip None values
    use vortex::dtype::{DType, Nullability};

    let with_nulls = VarBinArray::from_iter(
        vec![Some("foo"), None, Some("bar"), None, Some("baz")],
        DType::Utf8(Nullability::Nullable)
    );

    let non_null_strings = with_nulls.with_iterator(|iter| {
        iter.flatten()  // Skip None values
            .map(|bytes| unsafe { String::from_utf8_unchecked(bytes.to_vec()) })
            .collect::<Vec<_>>()
    }).expect("Failed to iterate");

    println!("Non-null strings only: {:?}", non_null_strings);

    // Count non-null values
    let count = with_nulls.with_iterator(|iter| {
        iter.filter(|opt| opt.is_some()).count()
    }).expect("Failed to count");

    println!("Non-null count: {}", count);

    // Transform strings using with_iterator
    let uppercased = varbin.with_iterator(|iter| {
        iter.map(|bytes_opt| {
            bytes_opt.map(|bytes| {
                let s = String::from_utf8(bytes.to_vec()).expect("Invalid UTF-8");
                s.to_uppercase()
            })
        })
        .collect::<Vec<_>>()
    }).expect("Failed to transform");

    println!("Uppercased strings: {:?}", uppercased);

    // Find strings matching a pattern
    let contains_an = varbin.with_iterator(|iter| {
        iter.enumerate()
            .filter_map(|(i, bytes_opt)| {
                bytes_opt.and_then(|bytes| {
                    let s = unsafe { String::from_utf8_unchecked(bytes.to_vec()) };
                    if s.contains("an") {
                        Some((i, s))
                    } else {
                        None
                    }
                })
            })
            .collect::<Vec<_>>()
    }).expect("Failed to search");

    println!("Strings containing 'an': {:?}", contains_an);
    // [array-accessor]

    // [array-iterator]
    // Use the array iterator for chunk-based iteration
    println!("\n=== Chunk-Based Iteration ===");

    use vortex::arrays::ChunkedArray;

    // Create a chunked array
    let chunk1 = buffer![1i32, 2, 3].into_array();
    let chunk2 = buffer![4i32, 5, 6].into_array();
    let chunked = ChunkedArray::from_iter([chunk1, chunk2]).into_array();

    println!("Chunked array: {}", chunked.display_values());

    // Iterate over chunks
    for (idx, chunk_result) in chunked.to_array_iterator().enumerate() {
        if let Ok(chunk) = chunk_result {
            println!("Chunk {}: {} elements", idx, chunk.len());
            println!("  Values: {}", chunk.display_values());
        }
    }
    // [array-iterator]

    // [manual-loop]
    // Process values in a loop
    println!("\n=== Processing Values ===");

    let numbers = buffer![1i32, 2, 3, 4, 5].into_array();
    let mut sum = 0i32;

    for i in 0..numbers.len() {
        if let Some(value) = numbers.scalar_at(i).as_primitive().typed_value::<i32>() {
            sum += value;
        }
    }

    println!("Sum of {}: {}", numbers.display_values(), sum);
    // [manual-loop]

    // [finding-values]
    // Search for specific values
    println!("\n=== Finding Values ===");

    let data = buffer![10i32, 20, 30, 20, 50].into_array();
    let target = 20i32;

    println!("Looking for {} in {}", target, data.display_values());
    for i in 0..data.len() {
        if let Some(value) = data.scalar_at(i).as_primitive().typed_value::<i32>()
            && value == target {
                println!("  Found at index {}", i);
            }
    }
    // [finding-values]

    // [modify-note]
    // Note: Arrays are immutable! To "modify", create a new array
    println!("\n=== Note on Mutability ===");
    println!("Vortex arrays are immutable.");
    println!("To create modified versions, use from_iter or collect:");

    // Use from_iter to create transformed array
    let transformed: Vec<i32> = (0..5).map(|i| i * 2).collect();
    let new_array = PrimitiveArray::from_iter(transformed);
    println!("New array: {}", new_array.display_values());
    // [modify-note]
}
