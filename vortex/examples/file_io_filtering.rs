//! File I/O and Filtering Example
//!
//! This example demonstrates how to write Vortex arrays to files and read them back
//! with filtering and predicate pushdown for efficient data access.
//!
//! Run with: cargo run --example file_io_filtering --features tokio

use vortex::arrays::{PrimitiveArray, StructArray, VarBinArray};
use vortex::compressor::CompactCompressor;
use vortex::dtype::{DType, Nullability};
use vortex::file::{VortexOpenOptions, VortexWriteOptions, WriteStrategyBuilder};
use vortex::stream::ArrayStreamExt;
use vortex::validity::Validity;
use vortex::{Array, ArrayLen, IntoArray};
use vortex_expr::operators::{gt, lt};
use vortex_expr::root;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Vortex File I/O and Filtering Example ===\n");

    // Create a temporary file
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("vortex_example.vortex");

    println!("1. Creating sample data...");
    let data = create_sample_data()?;
    println!("   Created {} records", data.len());

    println!("\n2. Writing data to file: {:?}", file_path);
    write_to_file(&file_path, data).await?;
    println!("   File written successfully!");

    // Check file size
    let file_size = std::fs::metadata(&file_path)?.len();
    println!(
        "   File size: {} bytes ({:.2} KB)",
        file_size,
        file_size as f64 / 1024.0
    );

    println!("\n3. Reading entire file...");
    read_entire_file(&file_path).await?;

    println!("\n4. Reading with filter (age > 30)...");
    read_with_filter(&file_path).await?;

    println!("\n5. Reading with complex filter (age > 25 AND age < 40)...");
    read_with_complex_filter(&file_path).await?;

    // Clean up
    std::fs::remove_file(&file_path)?;
    println!("\n=== Example completed successfully! ===");

    Ok(())
}

fn create_sample_data() -> Result<StructArray, Box<dyn std::error::Error>> {
    // Create a dataset of people with names, ages, and cities
    let names = vec![
        "Alice", "Bob", "Charlie", "Diana", "Eve", "Frank", "Grace", "Henry", "Iris", "Jack",
        "Kate", "Liam", "Mia", "Noah", "Olivia", "Peter", "Quinn", "Ruby", "Sam", "Tina",
    ];

    let ages = vec![
        25u32, 32, 28, 45, 22, 38, 29, 51, 27, 33, 41, 26, 36, 48, 24, 39, 30, 44, 35, 23,
    ];

    let cities = vec![
        "New York",
        "London",
        "Paris",
        "Tokyo",
        "Sydney",
        "Berlin",
        "Toronto",
        "Singapore",
        "Mumbai",
        "Dubai",
        "New York",
        "London",
        "Paris",
        "Tokyo",
        "Sydney",
        "Berlin",
        "Toronto",
        "Singapore",
        "Mumbai",
        "Dubai",
    ];

    let scores = vec![
        92.5f64, 88.3, 95.1, 76.8, 89.2, 84.7, 91.3, 78.5, 93.6, 87.9, 85.4, 90.2, 82.1, 79.6,
        94.3, 81.8, 88.7, 77.2, 86.5, 92.1,
    ];

    // Create arrays
    let name_array = VarBinArray::from_iter(names.iter(), DType::Utf8(Nullability::NonNullable));
    let age_array: PrimitiveArray = PrimitiveArray::from(ages);
    let city_array = VarBinArray::from_iter(cities.iter(), DType::Utf8(Nullability::NonNullable));
    let score_array: PrimitiveArray = PrimitiveArray::from(scores);

    // Create a struct array
    let struct_array = StructArray::try_new(
        ["name", "age", "city", "score"].into(),
        vec![
            name_array.into_array(),
            age_array.into_array(),
            city_array.into_array(),
            score_array.into_array(),
        ],
        names.len(),
        Validity::NonNullable,
    )?;

    Ok(struct_array)
}

async fn write_to_file(
    path: impl AsRef<std::path::Path>,
    data: StructArray,
) -> Result<(), Box<dyn std::error::Error>> {
    // Open file for writing
    let mut file = tokio::fs::File::create(path).await?;

    // Configure write options with compression
    let write_opts = VortexWriteOptions::default().with_strategy(
        WriteStrategyBuilder::new()
            .with_compressor(CompactCompressor::default())
            .build(),
    );

    // Write the data
    write_opts.write(&mut file, data.to_array_stream()).await?;

    Ok(())
}

async fn read_entire_file(
    path: impl AsRef<std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Open the file
    let reader = VortexOpenOptions::new().open(path).await?;

    // Create a scan operation
    let scan = reader.scan()?;

    // Read all data
    let array = scan.into_array_stream()?.read_all().await?;

    println!("   Read {} total rows", array.len());

    Ok(())
}

async fn read_with_filter(
    path: impl AsRef<std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Open the file
    let reader = VortexOpenOptions::new().open(path).await?;

    // Create a scan with filter: age > 30
    let age_field = root().field("age");
    let filter = gt(age_field, 30u64);

    let scan = reader.scan()?.with_filter(filter);

    // Read filtered data
    let array = scan.into_array_stream()?.read_all().await?;

    println!("   Read {} filtered rows", array.len());
    println!("   Note: Predicate pushdown allows reading only matching data from disk");

    Ok(())
}

async fn read_with_complex_filter(
    path: impl AsRef<std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    // Open the file
    let reader = VortexOpenOptions::new().open(path).await?;

    // Create a complex filter: age > 25 AND age < 40
    let age_field = root().field("age");
    let filter = gt(age_field.clone(), 25u64) & lt(age_field, 40u64);

    let scan = reader.scan()?.with_filter(filter);

    // Read filtered data
    let array = scan.into_array_stream()?.read_all().await?;

    println!("   Read {} filtered rows", array.len());
    println!("   Filter: 25 < age < 40");

    Ok(())
}
