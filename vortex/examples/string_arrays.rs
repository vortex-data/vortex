// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors


#![allow(clippy::expect_used)]
//! This example demonstrates working with string arrays in Vortex.
//!
//! Vortex has two main encodings for variable-length data like strings:
//! - VarBinArray: Uses offsets to index into a contiguous buffer (similar to Arrow's Utf8Array)
//! - VarBinViewArray: Uses views into one or more buffers (similar to Arrow's StringViewArray)
//!
//! VarBinViewArray is the canonical encoding for strings and is generally more efficient for
//! operations like slicing and concatenation.

use vortex::arrays::{VarBinArray, VarBinViewArray};
use vortex::dtype::{DType, Nullability};
use vortex::{Array, IntoArray};

fn main() {
    // [varbin-from-vec]
    // Create a VarBinArray from a Vec of string slices
    let varbin = VarBinArray::from(vec!["hello", "world", "vortex"]);
    println!("VarBin array: {}", varbin.display_values());
    // Output: ["hello", "world", "vortex"]
    // [varbin-from-vec]

    // [varbin-from-iter]
    // Create a VarBinArray from an iterator with a specific DType
    let nullable_strings = VarBinArray::from_iter(
        vec![Some("foo"), None, Some("bar"), Some("baz")],
        DType::Utf8(Nullability::Nullable),
    );
    println!("VarBin with nulls: {}", nullable_strings.display_values());
    // [varbin-from-iter]

    // [varbinview-from-vec]
    // Create a VarBinViewArray from a Vec of string slices
    // This is the canonical encoding for strings
    let view_array: VarBinViewArray = vec!["hello", "world", "vortex"]
        .into_iter()
        .map(Some)
        .collect();
    println!("VarBinView array: {}", view_array.display_values());
    // [varbinview-from-vec]

    // [varbinview-from-iter]
    // VarBinViewArray with nullable strings
    let nullable_views = VarBinViewArray::from_iter(
        vec![Some("alpha"), None, Some("beta"), Some("gamma")],
        DType::Utf8(Nullability::Nullable),
    );
    println!("VarBinView with nulls: {}", nullable_views.display_values());
    // [varbinview-from-iter]

    // [binary-data]
    // You can also create binary (non-UTF8) arrays
    let binary_data = VarBinArray::from_iter(
        vec![Some(b"binary".as_slice()), None, Some(b"data".as_slice())],
        DType::Binary(Nullability::Nullable),
    );
    println!("Binary array: {}", binary_data.display_values());
    // [binary-data]

    // [array-vs-view]
    // Understanding the difference between VarBinArray and VarBinViewArray
    println!("\n=== VarBinArray vs VarBinViewArray ===");

    // VarBinArray: Offset-based encoding (like Arrow StringArray)
    // Memory layout: [offsets: 0,5,18,19] [data: "shortmedium lengthx"]
    let varbin_arr = VarBinArray::from(vec!["short", "medium length", "x"]);
    println!("VarBinArray:");
    println!("  Encoding: {}", varbin_arr.encoding().id());
    println!("  Memory usage: {} bytes", varbin_arr.nbytes());
    println!("  Structure: Single data buffer + offsets");
    println!("  Best for: Sequential access, small uniform strings");

    // VarBinViewArray: View-based encoding (like Arrow StringViewArray)
    // Memory layout: views point to data in one or more buffers
    let view_arr: VarBinViewArray = vec!["short", "medium length", "x"]
        .into_iter()
        .map(Some)
        .collect();
    println!("\nVarBinViewArray:");
    println!("  Encoding: {}", view_arr.encoding().id());
    println!("  Memory usage: {} bytes", view_arr.nbytes());
    println!("  Structure: Multiple buffers + views");
    println!("  Best for: Slicing, concatenation, mixed-size strings");
    println!("  Is canonical: Yes (for Utf8 dtype)");
    // [array-vs-view]

    // [converting-between]
    // You can convert between encodings using to_canonical
    let varbin = VarBinArray::from(vec!["convert", "me"]).into_array();
    let canonical = varbin.to_canonical().into_array();
    println!(
        "\nConverted to canonical encoding: {}",
        canonical.encoding().id()
    );
    // Canonical for Utf8 is VarBinViewArray
    // [converting-between]

    // [empty-strings]
    // Empty strings are valid and different from null
    let with_empty = VarBinArray::from(vec!["", "not empty", ""]);
    println!(
        "\nArray with empty strings: {}",
        with_empty.display_values()
    );

    let with_nulls = VarBinArray::from_iter(
        vec![None, Some("not empty"), None],
        DType::Utf8(Nullability::Nullable),
    );
    println!("Array with nulls: {}", with_nulls.display_values());
    // [empty-strings]

    // [large-strings]
    // For very large strings (>4GB total), you can use the appropriate dtype
    let large_strings = VarBinArray::from(vec![
        "This example shows regular strings",
        "For datasets >4GB use appropriate offset types",
    ]);
    println!("\nStrings array dtype: {}", large_strings.dtype());
    // [large-strings]
}
