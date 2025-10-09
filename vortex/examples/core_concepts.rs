// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::expect_used)]

//! This example demonstrates core Vortex concepts like the Array trait vs concrete types.
//!
//! Understanding the difference between the Array trait and concrete array types is
//! fundamental to using Vortex effectively.

use vortex::arrays::{BoolArray, PrimitiveArray, VarBinArray};
use vortex::buffer::buffer;
use vortex::{Array, ArrayRef, IntoArray};

fn main() {
    // [array-trait-vs-concrete]
    // Array trait vs Concrete types demonstration
    println!("=== Array Trait vs Concrete Types ===\n");

    // Create specific concrete types
    let varbin = VarBinArray::from(vec!["hello", "world"]);
    let _primitive = PrimitiveArray::from_iter(vec![1i32, 2, 3]);

    // Both have type-specific methods
    let _bytes = varbin.bytes(); // VarBinArray-specific method
    println!("VarBinArray has {} bytes", varbin.bytes().len());

    // Convert to trait object (ArrayRef = Arc<dyn Array>)
    let array_ref: ArrayRef = varbin.into_array();

    // Now we can only use Array trait methods
    println!("Array trait methods work on any type:");
    println!("  Length: {}", array_ref.len());
    println!("  DType: {}", array_ref.dtype());
    println!("  Encoding: {}", array_ref.encoding().id());
    println!("  Values: {}", array_ref.display_values());

    // To use type-specific methods again, need to downcast
    if let Some(varbin_again) = array_ref.as_any().downcast_ref::<VarBinArray>() {
        println!("\nAfter downcast, can use VarBinArray methods:");
        println!("  Bytes length: {}", varbin_again.bytes().len());
        println!("  Offsets dtype: {}", varbin_again.offsets().dtype());
    }
    // [array-trait-vs-concrete]

    // [polymorphism]
    // Polymorphism: Write functions that work with any array type
    println!("\n=== Polymorphism ===\n");

    fn process_any_array(array: &dyn Array) {
        println!("Processing {} with {} elements", array.encoding().id(), array.len());
        println!("  First element: {}", array.scalar_at(0));
    }

    let int_array = buffer![10i32, 20, 30].into_array();
    let string_array = VarBinArray::from(vec!["foo", "bar"]).into_array();
    let bool_array: ArrayRef = BoolArray::from_iter([true, false]).into_array();

    process_any_array(int_array.as_ref());
    process_any_array(string_array.as_ref());
    process_any_array(bool_array.as_ref());
    // [polymorphism]

    // [heterogeneous-collections]
    // Heterogeneous collections: Store different array types together
    println!("\n=== Heterogeneous Collections ===\n");

    let arrays: Vec<ArrayRef> = vec![
        buffer![1, 2, 3].into_array(),                          // PrimitiveArray
        VarBinArray::from(vec!["a", "b"]).into_array(),         // VarBinArray
        BoolArray::from_iter([true, false]).into_array(),       // BoolArray
    ];

    println!("Collection of {} different array types:", arrays.len());
    for (i, array) in arrays.iter().enumerate() {
        println!(
            "  [{}] {} encoding with {} elements",
            i,
            array.encoding().id(),
            array.len()
        );
    }
    // [heterogeneous-collections]

    // [multiple-encodings]
    // Multiple encodings for same logical data
    println!("\n=== Multiple Encodings ===\n");

    use vortex::arrays::VarBinViewArray;

    // Same strings, different encodings
    let varbin = VarBinArray::from(vec!["hello", "world", "vortex"]);
    let view: VarBinViewArray = vec!["hello", "world", "vortex"]
        .into_iter()
        .map(Some)
        .collect();

    println!("Same data, different encodings:");
    println!("  VarBinArray encoding: {}", varbin.encoding().id());
    println!("  VarBinArray nbytes: {}", varbin.nbytes());
    println!("  VarBinViewArray encoding: {}", view.encoding().id());
    println!("  VarBinViewArray nbytes: {}", view.nbytes());

    // Both implement Array trait, can be used interchangeably
    let varbin_ref: ArrayRef = varbin.into_array();
    let view_ref: ArrayRef = view.into_array();

    // Both can be processed by the same function
    process_any_array(varbin_ref.as_ref());
    process_any_array(view_ref.as_ref());

    // Convert to canonical encoding
    let canonical = varbin_ref.to_canonical().into_array();
    println!("\nCanonical encoding: {}", canonical.encoding().id());
    // [multiple-encodings]
}