// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#![allow(clippy::expect_used)]

//! This example demonstrates additional array types in Vortex beyond the basics,
//! including ConstantArray, ChunkedArray, and NullArray.

use vortex::arrays::{ChunkedArray, ConstantArray, NullArray, VarBinArray};
use vortex::buffer::buffer;
use vortex::dtype::{DType, Nullability};
use vortex::scalar::Scalar;
use vortex::{Array, IntoArray};

fn main() {
    // [constant-array]
    println!("=== Constant Arrays ===\n");

    // ConstantArray: Efficiently represents arrays where all values are the same
    let constant = ConstantArray::new(Scalar::from(42i32), 1_000_000);

    println!("Constant array of 1M values:");
    println!("  Length: {}", constant.len());
    println!("  Memory usage: {} bytes (very efficient!)", constant.nbytes());
    println!("  First value: {}", constant.scalar_at(0));
    println!("  Last value: {}", constant.scalar_at(999_999));

    // Constant arrays are useful for default values or padding
    let zeros = ConstantArray::new(Scalar::from(0.0f64), 100);
    println!("\nArray of 100 zeros:");
    println!("  All values are: {}", zeros.scalar_at(0));
    // [constant-array]

    // [constant-null]
    // Constant null array
    use vortex::dtype::PType;
    let null_constant = ConstantArray::new(
        Scalar::null(DType::Primitive(PType::I32, Nullability::Nullable)),
        50
    );
    println!("\nConstant null array:");
    println!("  All values are null: {}", null_constant.scalar_at(0).is_null());
    // [constant-null]

    // [chunked-array]
    println!("\n=== Chunked Arrays ===\n");

    // ChunkedArray: Combines multiple arrays into a single logical array
    // Useful for streaming, parallel processing, or incremental data

    let chunk1 = buffer![1i32, 2, 3].into_array();
    let chunk2 = buffer![4i32, 5, 6].into_array();
    let chunk3 = buffer![7i32, 8, 9].into_array();

    let chunked = ChunkedArray::from_iter([chunk1, chunk2, chunk3]).into_array();

    println!("Chunked array:");
    println!("  Total length: {}", chunked.len());
    println!("  Values: {}", chunked.display_values());

    // Access individual elements (transparently across chunks)
    for i in 0..chunked.len() {
        println!("  Element {}: {}", i, chunked.scalar_at(i));
    }

    // You can also iterate over chunks
    println!("\nIterating over chunks:");
    for (idx, chunk_result) in chunked.to_array_iterator().enumerate() {
        if let Ok(chunk) = chunk_result {
            println!("  Chunk {}: {} elements", idx, chunk.len());
        }
    }
    // [chunked-array]

    // [mixed-type-chunks]
    // Chunks can be created from different sources
    println!("\n=== Mixed Source Chunks ===\n");

    let string_chunk1 = VarBinArray::from(vec!["hello", "world"]).into_array();
    let string_chunk2 = VarBinArray::from(vec!["foo", "bar", "baz"]).into_array();

    let string_chunked = ChunkedArray::from_iter([string_chunk1, string_chunk2]).into_array();
    println!("String chunks: {}", string_chunked.display_values());
    // [mixed-type-chunks]

    // [null-array]
    println!("\n=== Null Arrays ===\n");

    // NullArray: Arrays of all null values
    let nulls = NullArray::new(5);

    println!("NullArray:");
    println!("  Length: {}", nulls.len());
    println!("  Memory usage: {} bytes (minimal!)", nulls.nbytes());
    println!("  DType: {}", nulls.dtype());

    // All values are null
    for i in 0..nulls.len() {
        println!("  Element {}: {}", i, nulls.scalar_at(i));
    }

    // Useful for representing missing columns or as placeholders
    // [null-array]

    // [sparse-pattern]
    println!("\n=== Sparse-like Pattern with ConstantArray ===\n");

    // You can simulate sparse arrays by chunking constant arrays with actual data
    let default = ConstantArray::new(Scalar::from(0i32), 100).into_array();
    let actual_data = buffer![10i32, 20, 30].into_array();
    let more_defaults = ConstantArray::new(Scalar::from(0i32), 97).into_array();

    let sparse_like = ChunkedArray::from_iter([default, actual_data, more_defaults]).into_array();

    println!("Sparse-like array (200 elements, only 3 non-zero):");
    println!("  Total length: {}", sparse_like.len());
    println!("  Memory efficient for sparse data");
    // [sparse-pattern]
}