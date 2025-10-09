// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors


#![allow(clippy::expect_used)]
#![allow(clippy::expect_used)]
#![allow(clippy::use_debug)]
#![allow(unexpected_cfgs)]

//! This example demonstrates how to create and work with struct arrays.
//!
//! Struct arrays in Vortex are similar to structs in programming languages - they group
//! multiple fields together. Each field is itself an array, and all fields have the same length.

use vortex::arrays::{StructArray, VarBinArray};
use vortex::buffer::buffer;
use vortex::validity::Validity;
use vortex::{Array, IntoArray};

fn main() {
    // [struct-from-fields]
    // Create a struct array from field name/array pairs
    println!("=== Creating Struct Arrays ===");

    let names = VarBinArray::from(vec!["Alice", "Bob", "Charlie"]).into_array();
    let ages = buffer![30i32, 25, 35].into_array();

    let people = StructArray::from_fields(&[("name", names), ("age", ages)])
        .expect("Failed to create struct array")
        .into_array();

    println!("People struct: {}", people.display_values());
    // [struct-from-fields]

    // [struct-try-new]
    // Create a struct array with explicit validity
    println!("\n=== Struct with Validity ===");

    let x_values = buffer![1i32, 2, 3, 4].into_array();
    let y_values = buffer![10i32, 20, 30, 40].into_array();

    let validity: Validity = [true, true, false, true].into_iter().collect();
    let points = StructArray::try_new(
        ["x", "y"].into(),
        vec![x_values, y_values],
        4,        // length
        validity, // third point is null
    )
    .expect("Failed to create struct array with validity")
    .into_array();

    println!("Points: {}", points.display_values());
    // Output shows third struct as null
    // [struct-try-new]

    // [access-fields]
    // Access fields from a struct array
    println!("\n=== Accessing Fields ===");

    let struct_array = StructArray::from_fields(&[
        ("id", buffer![1i32, 2, 3].into_array()),
        (
            "label",
            VarBinArray::from(vec!["foo", "bar", "baz"]).into_array(),
        ),
    ])
    .expect("Failed to create struct array with id and label");

    // Get a specific field by name
    if let Ok(id_field) = struct_array.field_by_name("id") {
        println!("ID field: {}", id_field.display_values());
    }

    // Get field by index
    let label_field = &struct_array.fields()[1];
    println!("Label field: {}", label_field.display_values());

    // List all field names
    println!("Field names: {:?}", struct_array.names());
    // [access-fields]

    // [iterate-structs]
    // Iterate over struct values
    println!("\n=== Iterating Struct Values ===");

    let products = StructArray::from_fields(&[
        (
            "product",
            VarBinArray::from(vec!["Apple", "Banana", "Cherry"]).into_array(),
        ),
        ("price", buffer![1.20f64, 0.50, 2.00].into_array()),
        ("quantity", buffer![10i32, 25, 5].into_array()),
    ])
    .expect("Failed to create products struct array")
    .into_array();

    for i in 0..products.len() {
        let struct_scalar = products.scalar_at(i);
        println!("Row {}: {}", i, struct_scalar);

        // Access individual fields in the struct scalar
        let struct_val = struct_scalar.as_struct();
        if let Some(product) = struct_val.field("product") {
            println!("  Product: {}", product);
        }
    }
    // [iterate-structs]

    // [nested-structs]
    // Nested struct arrays
    println!("\n=== Nested Structs ===");

    // Create inner struct (address)
    let address = StructArray::from_fields(&[
        (
            "street",
            VarBinArray::from(vec!["123 Main St", "456 Oak Ave"]).into_array(),
        ),
        (
            "city",
            VarBinArray::from(vec!["Springfield", "Portland"]).into_array(),
        ),
    ])
    .expect("Failed to create address struct array")
    .into_array();

    // Create outer struct (person with address)
    let person_with_address = StructArray::from_fields(&[
        ("name", VarBinArray::from(vec!["Alice", "Bob"]).into_array()),
        ("address", address),
    ])
    .expect("Failed to create nested struct array")
    .into_array();

    println!("Nested structs: {}", person_with_address.display_values());
    // [nested-structs]

    // [struct-with-mixed-types]
    // Struct with various data types
    println!("\n=== Struct with Mixed Types ===");

    use vortex::arrays::BoolArray;

    let employees = StructArray::from_fields(&[
        ("id", buffer![1001u64, 1002, 1003].into_array()),
        (
            "name",
            VarBinArray::from(vec!["Alice", "Bob", "Charlie"]).into_array(),
        ),
        ("salary", buffer![75000.0f64, 82000.0, 95000.0].into_array()),
        (
            "active",
            BoolArray::from_iter([true, true, false]).into_array(),
        ),
    ])
    .expect("Failed to create employees struct array")
    .into_array();

    println!("Employees:");
    #[cfg(feature = "pretty")]
    println!("{}", employees.display_table());

    #[cfg(not(feature = "pretty"))]
    println!("{}", employees.display_values());
    // [struct-with-mixed-types]

    // [struct-properties]
    // Inspect struct properties
    println!("\n=== Struct Properties ===");

    let sample_struct = StructArray::from_fields(&[
        ("a", buffer![1i32, 2].into_array()),
        ("b", buffer![3i32, 4].into_array()),
    ])
    .expect("Failed to create sample struct array");

    println!("Number of fields: {}", sample_struct.names().len());
    println!("Length: {}", sample_struct.len());
    println!("Field names: {:?}", sample_struct.names());
    // [struct-properties]

    // [empty-struct]
    // Empty struct (struct with no fields)
    println!("\n=== Empty Struct ===");

    use vortex::dtype::FieldNames;

    let empty_struct = StructArray::try_new(
        FieldNames::empty(),
        vec![],
        3, // 3 rows, but no fields
        Validity::NonNullable,
    )
    .expect("Failed to create empty struct array")
    .into_array();

    println!("Empty struct (3 rows): {}", empty_struct.display_values());
    // [empty-struct]
}
