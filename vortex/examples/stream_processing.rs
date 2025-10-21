//! Stream Processing Example
//!
//! This example demonstrates how to process large datasets efficiently using
//! Vortex's chunked arrays and streaming capabilities.
//!
//! Use case: Processing sensor data in a streaming fashion
//!
//! Run with: cargo run --example stream_processing --features tokio

use vortex::arrays::{ChunkedArray, PrimitiveArray, StructArray};
use vortex::compressor::CompactCompressor;
use vortex::dtype::{DType, Nullability, PType};
use vortex::file::{VortexOpenOptions, VortexWriteOptions, WriteStrategyBuilder};
use vortex::stream::ArrayStreamExt;
use vortex::validity::Validity;
use vortex::{Array, ArrayLen, IntoArray, IntoCanonical};
use vortex_expr::operators::gt;
use vortex_expr::root;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Vortex Stream Processing Example ===\n");
    println!("Simulating real-time sensor data processing with chunked arrays.\n");

    // Step 1: Generate sensor data in chunks
    println!("Step 1: Generating sensor data in chunks...");
    let sensor_data = generate_sensor_data_chunked(10_000, 1_000)?;
    let chunked_data = sensor_data
        .clone()
        .into_canonical()?
        .into_chunked()
        .ok_or("Expected chunked")?;
    println!(
        "   Generated {} total readings in {} chunks",
        sensor_data.len(),
        chunked_data.nchunks()
    );

    // Step 2: Process each chunk individually (memory efficient)
    println!("\nStep 2: Processing chunks individually...");
    process_chunks_sequentially(&chunked_data)?;

    // Step 3: Write chunked data to file
    println!("\nStep 3: Writing chunked data to file...");
    let temp_dir = std::env::temp_dir();
    let file_path = temp_dir.join("sensor_data.vortex");
    write_chunked_data(&file_path, &sensor_data).await?;

    let file_size = std::fs::metadata(&file_path)?.len();
    println!(
        "   File written: {} bytes ({:.2} KB)",
        file_size,
        file_size as f64 / 1024.0
    );

    // Step 4: Stream data from file in chunks
    println!("\nStep 4: Streaming data from file...");
    stream_from_file(&file_path).await?;

    // Step 5: Process with filtering while streaming
    println!("\nStep 5: Streaming with filter (temperature > 25.0)...");
    stream_with_filter(&file_path).await?;

    // Clean up
    std::fs::remove_file(&file_path)?;

    println!("\n=== Stream processing completed successfully! ===");
    Ok(())
}

/// Generate sensor data in chunks (simulating real-time ingestion)
fn generate_sensor_data_chunked(
    total_readings: usize,
    chunk_size: usize,
) -> Result<ChunkedArray, Box<dyn std::error::Error>> {
    use rand::Rng;
    let mut rng = rand::thread_rng();

    let num_chunks = (total_readings + chunk_size - 1) / chunk_size;
    let mut chunks = Vec::new();

    for chunk_idx in 0..num_chunks {
        let start = chunk_idx * chunk_size;
        let end = std::cmp::min(start + chunk_size, total_readings);
        let current_chunk_size = end - start;

        // Generate timestamps (sequential)
        let timestamps: PrimitiveArray =
            (start..end).map(|i| (1700000000 + i * 60) as i64).collect();

        // Generate sensor IDs (cycling through 10 sensors)
        let sensor_ids: PrimitiveArray = (start..end).map(|i| (i % 10) as u32).collect();

        // Generate temperature readings (random with trend)
        let temperatures: PrimitiveArray = (start..end)
            .map(|i| {
                let base_temp = 20.0 + (i as f64 / 1000.0).sin() * 10.0;
                base_temp + rng.gen_range(-2.0..2.0)
            })
            .collect();

        // Generate humidity readings
        let humidity: PrimitiveArray = (0..current_chunk_size)
            .map(|_| rng.gen_range(30.0..90.0))
            .collect();

        // Create chunk as struct array
        let chunk = StructArray::try_new(
            ["timestamp", "sensor_id", "temperature", "humidity"].into(),
            vec![
                timestamps.into_array(),
                sensor_ids.into_array(),
                temperatures.into_array(),
                humidity.into_array(),
            ],
            current_chunk_size,
            Validity::NonNullable,
        )?;

        chunks.push(chunk.into_array());
    }

    // Create a chunked array
    let chunked = ChunkedArray::try_new(
        chunks,
        DType::Struct(
            vortex_dtype::StructDType::new(
                ["timestamp", "sensor_id", "temperature", "humidity"].into(),
                vec![
                    DType::Primitive(PType::I64, Nullability::NonNullable),
                    DType::Primitive(PType::U32, Nullability::NonNullable),
                    DType::Primitive(PType::F64, Nullability::NonNullable),
                    DType::Primitive(PType::F64, Nullability::NonNullable),
                ],
            ),
            Nullability::NonNullable,
        ),
    )?;

    Ok(chunked)
}

fn process_chunks_sequentially(chunked: &ChunkedArray) -> Result<(), Box<dyn std::error::Error>> {
    let mut total_readings = 0;
    let mut sum_temperature = 0.0;
    let mut min_temp = f64::MAX;
    let mut max_temp = f64::MIN;

    println!("   Processing {} chunks...", chunked.nchunks());

    for (chunk_idx, chunk) in chunked.chunks().enumerate() {
        let chunk_canonical = chunk.into_canonical()?;
        let chunk_struct = chunk_canonical
            .into_struct()
            .ok_or("Expected struct array")?;

        let temp_field = chunk_struct.field_by_name("temperature")?;
        let temps_canonical = temp_field.into_canonical()?;
        let temps = temps_canonical
            .into_primitive()
            .ok_or("Expected primitive array")?;

        // Process this chunk
        for i in 0..temps.len() {
            let temp = temps.get_as::<f64>(i).ok_or("Invalid temperature")?;
            sum_temperature += temp;
            min_temp = min_temp.min(temp);
            max_temp = max_temp.max(temp);
            total_readings += 1;
        }

        if chunk_idx % 3 == 0 {
            println!(
                "     Processed chunk {} ({} readings so far)",
                chunk_idx, total_readings
            );
        }
    }

    let avg_temperature = sum_temperature / total_readings as f64;

    println!("\n   Statistics across all chunks:");
    println!("     Total readings: {}", total_readings);
    println!("     Average temperature: {:.2}°C", avg_temperature);
    println!("     Min temperature: {:.2}°C", min_temp);
    println!("     Max temperature: {:.2}°C", max_temp);

    Ok(())
}

async fn write_chunked_data(
    path: impl AsRef<std::path::Path>,
    data: &ChunkedArray,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = tokio::fs::File::create(path).await?;

    let write_opts = VortexWriteOptions::default().with_strategy(
        WriteStrategyBuilder::new()
            .with_compressor(CompactCompressor::default())
            .build(),
    );

    // Write maintains the chunked structure
    write_opts
        .write(&mut file, data.clone().to_array_stream())
        .await?;

    Ok(())
}

async fn stream_from_file(
    path: impl AsRef<std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = VortexOpenOptions::new().open(path).await?;

    let scan = reader.scan()?;
    let array = scan.into_array_stream()?.read_all().await?;

    println!("   Streamed all data: {} readings", array.len());
    println!("   Note: Data is read incrementally, not all at once");

    Ok(())
}

async fn stream_with_filter(
    path: impl AsRef<std::path::Path>,
) -> Result<(), Box<dyn std::error::Error>> {
    let reader = VortexOpenOptions::new().open(path).await?;

    // Create filter for high temperatures
    let temp_filter = gt(root().field("temperature"), 25.0f64);

    let scan = reader.scan()?.with_filter(temp_filter);
    let array = scan.into_array_stream()?.read_all().await?;

    println!("   High temperature readings (> 25°C): {}", array.len());
    println!("   Note: Filter applied during read, reducing data transfer");

    Ok(())
}
