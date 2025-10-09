// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::expect_used)]
#![allow(clippy::use_debug)]
//! This example demonstrates different ways to print and inspect Vortex arrays.
//!
//! Vortex provides several display options for debugging and inspecting arrays:
//! - display_values(): Show logical values
//! - display_tree(): Show encoding tree structure
//! - display_table(): Show data in table format (when table-display feature is enabled)
//! - Default Display: Show metadata only

use vortex::arrays::{StructArray, VarBinArray};
use vortex::buffer::buffer;
use vortex::display::DisplayOptions;
use vortex::{Array, IntoArray};

fn main() {
    let int_array = buffer![1i32, 2, 3, 4, 5].into_array();

    // [default-display]
    // Default display shows encoding and metadata only
    println!("=== Default Display (Metadata) ===");
    println!("{}", int_array);
    // [default-display]

    // [display-values]
    // Display logical values of the array
    println!("\n=== Display Values ===");
    println!("{}", int_array.display_values());
    // [display-values]

    // [display-tree]
    // Display the encoding tree structure with memory info
    // Shows the internal structure, encodings, buffers, and memory usage
    println!("\n=== Display Tree ===");
    println!("{}", int_array.display_tree());
    // [display-tree]

    // [metadata-only]
    // Explicitly use metadata-only display
    println!("\n=== Metadata Only ===");
    println!("{}", int_array.display_as(DisplayOptions::MetadataOnly));
    // [metadata-only]

    // [complex-array]
    // For more complex arrays, the tree display is very useful
    println!("\n=== Complex Array Structure ===");

    let struct_array = StructArray::from_fields(&[
        ("numbers", buffer![10i32, 20, 30].into_array()),
        (
            "strings",
            VarBinArray::from(vec!["foo", "bar", "baz"]).into_array(),
        ),
    ])
    .expect("struct array should be instantiated from Buffer and VarBinArray")
    .into_array();

    println!("Struct values:");
    println!("{}", struct_array.display_values());

    println!("\nStruct tree:");
    println!("{}", struct_array.display_tree());
    // [complex-array]

    // [table-display]
    // Table display is great for struct arrays (requires pretty feature)
    #[cfg(feature = "pretty")]
    {
        println!("\n=== Table Display ===");
        println!("{}", struct_array.display_table());
        // Displays data in a nicely formatted table
    }
    // [table-display]

    // [inspect-properties]
    // You can also inspect individual properties programmatically
    println!("\n=== Inspecting Array Properties ===");
    println!("Length: {}", int_array.len());
    println!("DType: {}", int_array.dtype());
    println!("Encoding ID: {}", int_array.encoding_id());
    println!("Encoding: {}", int_array.encoding().id());
    println!("Is canonical: {}", int_array.is_canonical());
    println!("Bytes in memory: {}", int_array.nbytes());
    // [inspect-properties]

    // [inspect-validity]
    // Check validity for specific indices
    println!("\n=== Checking Validity ===");
    use vortex::arrays::PrimitiveArray;
    use vortex::validity::Validity;

    let validity: Validity = [true, false, true].into_iter().collect();
    let nullable_array = PrimitiveArray::new(buffer![1i32, 2, 3], validity);

    println!("Array: {}", nullable_array.display_values());
    for i in 0..nullable_array.len() {
        println!(
            "  Index {}: {} (valid: {})",
            i,
            nullable_array.scalar_at(i),
            nullable_array.is_valid(i)
        );
    }
    // [inspect-validity]

    // [debug-trait]
    // Arrays also implement Debug trait for use with debugging macros
    println!("\n=== Debug Trait ===");
    println!("Debug output: {:?}", int_array.dtype());
    // [debug-trait]

    // [statistics]
    // You can inspect array statistics
    println!("\n=== Array Statistics ===");

    let stats_array = buffer![5i32, 1, 9, 3, 7].into_array();
    println!("Array: {}", stats_array.display_values());

    // Arrays store statistics that can be queried
    // Note: Individual stats like min/max are available via the stats module
    // but require decompressing to canonical form for arbitrary encodings
    println!("Nbytes: {}", stats_array.nbytes());
    // [statistics]
}
