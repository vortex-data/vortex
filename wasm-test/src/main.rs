// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! WASI integration test for Vortex.
//!
//! This binary is compiled to `wasm32-wasip1` and executed via Wasmer to verify that Vortex works
//! correctly in a WASI environment.

use vortex::Array;
use vortex::IntoArray;
use vortex::arrays::{ConstantArray, PrimitiveArray, StructArray};
use vortex::buffer::{Buffer, buffer};
use vortex::compressor::BtrBlocksCompressor;
use vortex::compute::{Operator, compare, take};
use vortex::scalar::Scalar;
use vortex::validity::Validity;

fn main() {
    println!("Running Vortex WASI integration tests...\n");

    test_primitive_array();
    test_compute_operations();
    test_encodings();
    test_compression();
    test_array_types();

    println!("\nAll WASI integration tests passed!");
}

fn test_primitive_array() {
    println!("Testing PrimitiveArray creation...");

    let data: Vec<i32> = (0..1000).collect();
    let buffer = Buffer::from(data);
    let array = PrimitiveArray::new(buffer, Validity::NonNullable);

    assert_eq!(array.len(), 1000);
    println!("  Created PrimitiveArray with {} elements", array.len());
}

fn test_compute_operations() {
    println!("Testing compute operations...");

    let data: Vec<i32> = vec![1, 2, 3, 4, 5];
    let buffer = Buffer::from(data);
    let array = PrimitiveArray::new(buffer, Validity::NonNullable).into_array();

    // Test comparison.
    let threshold_array = ConstantArray::new(Scalar::from(3i32), 5).into_array();
    let comparison = compare(&array, &threshold_array, Operator::Gt).expect("compare failed");
    assert_eq!(comparison.len(), 5);
    println!("  Comparison operation succeeded");

    // Test take.
    let indices: Vec<u64> = vec![0, 2, 4];
    let indices_buffer = Buffer::from(indices);
    let indices_array = PrimitiveArray::new(indices_buffer, Validity::NonNullable).into_array();
    let taken = take(&array, &indices_array).expect("take failed");
    assert_eq!(taken.len(), 3);
    println!("  Take operation succeeded");
}

fn test_encodings() {
    println!("Testing encoding types...");

    use vortex::encodings;

    // Verify encodings are linked by checking their sizes.
    let alp_size = std::mem::size_of::<encodings::alp::ALPArray>();
    let bitpacked_size = std::mem::size_of::<encodings::fastlanes::BitPackedArray>();
    let runend_size = std::mem::size_of::<encodings::runend::RunEndArray>();
    let zigzag_size = std::mem::size_of::<encodings::zigzag::ZigZagArray>();

    assert!(alp_size > 0);
    assert!(bitpacked_size > 0);
    assert!(runend_size > 0);
    assert!(zigzag_size > 0);

    println!("  ALP, BitPacked, RunEnd, ZigZag encodings are linked");
}

fn test_compression() {
    println!("Testing compression...");

    // Create an array with repeated values (good for compression).
    let array = PrimitiveArray::new(buffer![1i32; 1024], Validity::AllValid).to_array();
    let original_len = array.len();

    let compressed = BtrBlocksCompressor::default()
        .compress(&array)
        .expect("compression failed");

    println!(
        "  Compressed array: {} -> {} elements",
        original_len,
        compressed.len()
    );
}

fn test_array_types() {
    println!("Testing array types...");

    // ConstantArray.
    let const_array = ConstantArray::new(Scalar::from(42i32), 100);
    assert_eq!(const_array.len(), 100);
    println!("  ConstantArray created");

    // StructArray.
    let field1 = PrimitiveArray::new(Buffer::from(vec![1i32, 2, 3]), Validity::NonNullable);
    let field2 = PrimitiveArray::new(Buffer::from(vec![4i32, 5, 6]), Validity::NonNullable);
    let struct_array =
        StructArray::from_fields(&[("a", field1.into_array()), ("b", field2.into_array())])
            .expect("StructArray creation failed");
    assert_eq!(struct_array.len(), 3);
    println!("  StructArray created with 2 fields");

    // Different numeric types.
    let int_array = PrimitiveArray::new(Buffer::from(vec![1i64, 2, 3, 4]), Validity::NonNullable);
    let float_array =
        PrimitiveArray::new(Buffer::from(vec![1.0f64, 2.0, 3.0]), Validity::NonNullable);
    assert_eq!(int_array.len(), 4);
    assert_eq!(float_array.len(), 3);
    println!("  PrimitiveArrays with i64 and f64 created");
}
