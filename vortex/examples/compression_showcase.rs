// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression Strategies Showcase
//!
//! This example demonstrates Vortex's powerful compression capabilities,
//! comparing different encoding strategies for various data patterns.
//!
//! Run with: cargo run --example compression_showcase

use vortex::array::ArrayRef;
use vortex::array::IntoArray;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::VarBinArray;
use vortex::array::validity::Validity;
use vortex::compressor::BtrBlocksCompressor;
use vortex::dtype::DType;
use vortex::dtype::Nullability;
use vortex_buffer::Buffer;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Vortex Compression Showcase ===\n");

    println!("This example demonstrates how Vortex automatically selects");
    println!("optimal compression strategies for different data patterns.\n");

    // 1. Compress sequential/monotonic data
    println!("1. Sequential Data Compression:");
    compress_sequential_data()?;

    // 2. Compress repetitive data
    println!("\n2. Repetitive Data Compression:");
    compress_repetitive_data()?;

    // 3. Compress string data
    println!("\n3. String Data Compression:");
    compress_string_data()?;

    // 4. Compress floating-point data
    println!("\n4. Floating-Point Data Compression:");
    compress_float_data()?;

    // 5. Compress sparse data
    println!("\n5. Sparse Data Compression:");
    compress_sparse_data()?;

    // 6. Compress structured data
    println!("\n6. Structured Data Compression:");
    compress_structured_data()?;

    println!("\n=== Compression showcase completed! ===");
    Ok(())
}

fn compress_sequential_data() -> Result<(), Box<dyn std::error::Error>> {
    // Create sequential data (e.g., timestamps, IDs)
    let sequential: PrimitiveArray = (1000..11000).map(|i| i as u64).collect();

    let uncompressed_size = estimate_size(&sequential.clone().into_array());
    println!("  Original sequential data (10,000 values):");
    println!("    Uncompressed size: ~{} bytes", uncompressed_size);

    // Compress using default strategy
    let compressor = BtrBlocksCompressor::default();
    let compressed = compressor.compress(&sequential.into_array())?;

    let compressed_size = compressed.nbytes();
    let ratio = uncompressed_size as f64 / compressed_size as f64;

    println!("    Compressed size: ~{} bytes", compressed_size);
    println!("    Compression ratio: {:.2}x", ratio);
    println!("    Encoding: {}", compressed.encoding_id());
    println!("    Note: Sequential data often compresses well with Delta or FoR encoding");

    Ok(())
}

fn compress_repetitive_data() -> Result<(), Box<dyn std::error::Error>> {
    // Create highly repetitive data (run-length encoding opportunity)
    let mut repetitive = Vec::new();
    for i in 0..100 {
        for _ in 0..100 {
            repetitive.push(i as u32);
        }
    }
    let array: PrimitiveArray = repetitive.into_iter().collect();

    let uncompressed_size = estimate_size(&array.clone().into_array());
    println!("  Repetitive data (100 values, each repeated 100 times):");
    println!("    Uncompressed size: ~{} bytes", uncompressed_size);

    let compressor = BtrBlocksCompressor::default();
    let compressed = compressor.compress(&array.into_array())?;

    let compressed_size = compressed.nbytes();
    let ratio = uncompressed_size as f64 / compressed_size as f64;

    println!("    Compressed size: ~{} bytes", compressed_size);
    println!("    Compression ratio: {:.2}x", ratio);
    println!("    Encoding: {}", compressed.encoding_id());
    println!("    Note: RLE (Run-Length Encoding) is ideal for repetitive data");

    Ok(())
}

fn compress_string_data() -> Result<(), Box<dyn std::error::Error>> {
    // Create string data with patterns
    let categories = vec!["Electronics", "Clothing", "Food", "Books"];
    let mut strings = Vec::new();

    // Repeat categories multiple times (good for dictionary encoding)
    for _ in 0..2500 {
        for category in &categories {
            strings.push(Some(*category));
        }
    }

    let array = VarBinArray::from_iter(strings, DType::Utf8(Nullability::NonNullable));

    let uncompressed_size = estimate_size(&array.clone().into_array());
    println!("  Categorical string data (10,000 strings, 4 categories):");
    println!("    Uncompressed size: ~{} bytes", uncompressed_size);

    let compressor = BtrBlocksCompressor::default();
    let compressed = compressor.compress(&array.into_array())?;

    let compressed_size = compressed.nbytes();
    let ratio = uncompressed_size as f64 / compressed_size as f64;

    println!("    Compressed size: ~{} bytes", compressed_size);
    println!("    Compression ratio: {:.2}x", ratio);
    println!("    Encoding: {}", compressed.encoding_id());
    println!("    Note: Dictionary encoding is excellent for categorical/repetitive strings");

    Ok(())
}

fn compress_float_data() -> Result<(), Box<dyn std::error::Error>> {
    // Create floating-point data with patterns
    let floats: Buffer<f64> = (0..10000).map(|i| (i as f64) * 0.1 + 100.0).collect();
    let array = floats.into_array();

    let uncompressed_size = estimate_size(&array);
    println!("  Floating-point data (10,000 values):");
    println!("    Uncompressed size: ~{} bytes", uncompressed_size);

    let compressor = BtrBlocksCompressor::default();
    let compressed = compressor.compress(&array)?;

    let compressed_size = compressed.nbytes();
    let ratio = uncompressed_size as f64 / compressed_size as f64;

    println!("    Compressed size: ~{} bytes", compressed_size);
    println!("    Compression ratio: {:.2}x", ratio);
    println!("    Encoding: {}", compressed.encoding_id());
    println!("    Note: ALP or PCO encodings are optimized for floating-point data");

    Ok(())
}

fn compress_sparse_data() -> Result<(), Box<dyn std::error::Error>> {
    // Create sparse data (mostly zeros with few non-zero values)
    let mut sparse = vec![0i64; 10000];
    for i in (0..10000).step_by(100) {
        sparse[i] = (i * 42) as i64;
    }
    let array: PrimitiveArray = sparse.into_iter().collect();

    let uncompressed_size = estimate_size(&array.clone().into_array());
    println!("  Sparse data (10,000 values, 99% zeros):");
    println!("    Uncompressed size: ~{} bytes", uncompressed_size);

    let compressor = BtrBlocksCompressor::default();
    let compressed = compressor.compress(&array.into_array())?;

    let compressed_size = compressed.nbytes();
    let ratio = uncompressed_size as f64 / compressed_size as f64;

    println!("    Compressed size: ~{} bytes", compressed_size);
    println!("    Compression ratio: {:.2}x", ratio);
    println!("    Encoding: {}", compressed.encoding_id());
    println!("    Note: Sparse encoding stores only non-zero indices and values");

    Ok(())
}

fn compress_structured_data() -> Result<(), Box<dyn std::error::Error>> {
    // Create a struct array with multiple columns
    let size = 5000;

    // ID column (sequential)
    let ids: PrimitiveArray = (1..=size).map(|i| i as u64).collect();

    // Status column (categorical)
    let statuses: Vec<Option<&str>> = (0..size)
        .map(|i| match i % 3 {
            0 => "active",
            1 => "pending",
            _ => "completed",
        })
        .map(Some)
        .collect();
    let status_array = VarBinArray::from_iter(statuses, DType::Utf8(Nullability::NonNullable));

    // Value column (floats)
    let values: PrimitiveArray = (0..size).map(|i| (i as f64) * 1.5).collect();

    let struct_array = StructArray::try_new(
        ["id", "status", "value"].into(),
        vec![
            ids.into_array(),
            status_array.into_array(),
            values.into_array(),
        ],
        size,
        Validity::NonNullable,
    )?;

    let uncompressed_size = estimate_size(&struct_array.clone().into_array());
    println!("  Structured data (5,000 records, 3 columns):");
    println!("    Uncompressed size: ~{} bytes", uncompressed_size);

    let compressor = BtrBlocksCompressor::default();
    let compressed = compressor.compress(&struct_array.into_array())?;

    let compressed_size = compressed.nbytes();
    let ratio = uncompressed_size as f64 / compressed_size as f64;

    println!("    Compressed size: ~{} bytes", compressed_size);
    println!("    Compression ratio: {:.2}x", ratio);
    println!("    Encoding: {}", compressed.encoding_id());
    println!("    Note: Each column can be compressed with its optimal strategy");

    Ok(())
}

/// Estimate the size of an array in bytes (approximation)
#[allow(clippy::cast_possible_truncation)]
fn estimate_size(array: &ArrayRef) -> usize {
    array.nbytes() as usize
}
