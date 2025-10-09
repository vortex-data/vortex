// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors


#![allow(clippy::expect_used)]
#![allow(clippy::unwrap_used)]
#![allow(clippy::use_debug)]
#![allow(unexpected_cfgs)]

//! This example demonstrates how to read and write Vortex files.
//!
//! Vortex provides async file I/O using Tokio. Files can be written with various
//! compression strategies and read with filtering and projection capabilities.

use vortex::arrays::{PrimitiveArray, StructArray, VarBinArray};
use vortex::buffer::buffer;
use vortex::stream::ArrayStreamExt;
use vortex::validity::Validity;
use vortex::{Array, IntoArray, ToCanonical};
use vortex_error::VortexResult;
use vortex_expr::{gt, lit, root};
use vortex_file::{VortexOpenOptions, VortexWriteOptions, WriteStrategyBuilder};
use vortex_layout::layouts::compact::CompactCompressor;

#[tokio::main]
async fn main() -> VortexResult<()> {
    basic_write_read().await?;
    compressed_write_read().await?;
    filtered_read().await?;
    struct_write_read().await?;

    // Cleanup
    let _ = tokio::fs::remove_file("example_basic.vortex").await;
    let _ = tokio::fs::remove_file("example_compressed.vortex").await;
    let _ = tokio::fs::remove_file("example_filtered.vortex").await;
    let _ = tokio::fs::remove_file("example_struct.vortex").await;

    Ok(())
}

async fn basic_write_read() -> VortexResult<()> {
    println!("=== Basic Write and Read ===\n");

    // [basic-write]
    // Create an array
    let array = PrimitiveArray::new(buffer![0u64, 1, 2, 3, 4], Validity::NonNullable);

    // Write to file using default options
    VortexWriteOptions::default()
        .write(
            &mut tokio::fs::File::create("example_basic.vortex").await?,
            array.to_array_stream(),
        )
        .await?;

    println!("Written array: {}", array.display_values());
    // [basic-write]

    // [basic-read]
    // Read the entire file back
    let read_array = VortexOpenOptions::new()
        .open("example_basic.vortex")
        .await?
        .scan()?
        .into_array_stream()?
        .read_all()
        .await?;

    println!("Read array: {}", read_array.display_values());
    println!("Arrays match: {}\n", array.len() == read_array.len());
    // [basic-read]

    Ok(())
}

async fn compressed_write_read() -> VortexResult<()> {
    println!("=== Compressed Write and Read ===\n");

    // [compressed-write]
    let array = buffer![42u64; 10000].into_array();

    println!("Original array nbytes: {}", array.nbytes());

    // Write with compact compression
    VortexWriteOptions::default()
        .with_strategy(
            WriteStrategyBuilder::new()
                .with_compressor(CompactCompressor::default())
                .build(),
        )
        .write(
            &mut tokio::fs::File::create("example_compressed.vortex").await?,
            array.to_array_stream(),
        )
        .await?;

    let file_size = tokio::fs::metadata("example_compressed.vortex")
        .await?
        .len();
    println!("File size: {} bytes", file_size);
    println!(
        "Compression ratio: {:.2}x\n",
        array.nbytes() as f64 / file_size as f64
    );
    // [compressed-write]

    // [compressed-read]
    let read_array = VortexOpenOptions::new()
        .open("example_compressed.vortex")
        .await?
        .scan()?
        .into_array_stream()?
        .read_all()
        .await?;

    println!("Read compressed array length: {}", read_array.len());
    // [compressed-read]

    Ok(())
}

async fn filtered_read() -> VortexResult<()> {
    println!("=== Filtered Read (Pushdown) ===\n");

    // [filtered-write]
    // Write an array with values 0-99
    let array = PrimitiveArray::from_iter(0..100u64);

    VortexWriteOptions::default()
        .write(
            &mut tokio::fs::File::create("example_filtered.vortex").await?,
            array.to_array_stream(),
        )
        .await?;
    // [filtered-write]

    // [filtered-read]
    // Read only values greater than 50
    let filtered = VortexOpenOptions::new()
        .open("example_filtered.vortex")
        .await?
        .scan()?
        .with_filter(gt(root(), lit(50u64)))
        .into_array_stream()?
        .read_all()
        .await?;

    println!("Original length: {}", array.len());
    println!("Filtered length: {}", filtered.len());
    println!(
        "Filtered values (first 10): {:?}",
        (0..10.min(filtered.len()))
            .map(|i| filtered.scalar_at(i).as_primitive().typed_value::<u64>())
            .collect::<Vec<_>>()
    );
    // [filtered-read]

    Ok(())
}

async fn struct_write_read() -> VortexResult<()> {
    println!("\n=== Struct Write and Read ===\n");

    // [struct-write]
    // Create a struct array with multiple fields
    let names = VarBinArray::from(vec!["Alice", "Bob", "Charlie", "Diana"]).into_array();
    let ages = buffer![30i32, 25, 35, 28].into_array();
    let scores = buffer![95.5f64, 87.3, 91.2, 88.9].into_array();

    let people = StructArray::from_fields(&[("name", names), ("age", ages), ("score", scores)])
        .unwrap()
        .into_array();

    println!("Writing struct array:");
    #[cfg(feature = "pretty")]
    println!("{}", people.display_table());
    #[cfg(not(feature = "pretty"))]
    println!("{}", people.display_values());

    VortexWriteOptions::default()
        .write(
            &mut tokio::fs::File::create("example_struct.vortex").await?,
            people.to_array_stream(),
        )
        .await?;
    // [struct-write]

    // [struct-read]
    let read_struct = VortexOpenOptions::new()
        .open("example_struct.vortex")
        .await?
        .scan()?
        .into_array_stream()?
        .read_all()
        .await?;

    println!("\nRead struct array:");
    #[cfg(feature = "pretty")]
    println!("{}", read_struct.display_table());
    #[cfg(not(feature = "pretty"))]
    println!("{}", read_struct.display_values());
    // [struct-read]

    // [field-access]
    // Access specific fields after reading
    let struct_arr = read_struct.to_struct();
    if let Ok(age_field) = struct_arr.field_by_name("age") {
        println!("\nAges only: {}", age_field.display_values());
    }
    // [field-access]

    Ok(())
}
