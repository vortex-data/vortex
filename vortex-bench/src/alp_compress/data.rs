// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Synthetic data generation for the ALP compression benchmark.
//!
//! Generates f64 columns whose values use full floating-point precision,
//! ensuring the Vortex compressor selects ALP-RD encoding rather than
//! regular ALP.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::Float64Array;
use arrow_array::Int64Array;
use arrow_array::RecordBatch;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use parquet::arrow::ArrowWriter;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;

/// Batch size for writing Parquet.
const BATCH_SIZE: usize = 100_000;

/// Generate the `alp_floats` Parquet file with synthetic f64 columns that
/// exercise ALP-RD compression.
pub fn generate_alp_floats_parquet(n_rows: usize, path: &Path) -> Result<()> {
    let schema = alp_floats_schema();
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;
    let mut rng = StdRng::seed_from_u64(42);

    for batch_start in (0..n_rows).step_by(BATCH_SIZE) {
        let batch_len = BATCH_SIZE.min(n_rows - batch_start);
        let batch = generate_batch(&schema, batch_start, batch_len, &mut rng);
        writer.write(&batch)?;
    }

    writer.close()?;
    Ok(())
}

/// Schema for the `alp_floats` table.
pub fn alp_floats_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("sensor_reading", DataType::Float64, false),
        Field::new("price", DataType::Float64, false),
        Field::new("measurement", DataType::Float64, false),
        Field::new("temperature", DataType::Float64, false),
        Field::new("velocity", DataType::Float64, false),
        Field::new("label", DataType::Int64, false),
    ]))
}

fn generate_batch(
    schema: &Arc<Schema>,
    batch_start: usize,
    batch_len: usize,
    rng: &mut StdRng,
) -> RecordBatch {
    let ids =
        Int64Array::from_iter_values((batch_start as i64)..((batch_start + batch_len) as i64));

    // sensor_reading: body-temperature-like values with sub-degree noise.
    // Shared exponent around 98.6, full mantissa variation.
    let sensor_reading = Float64Array::from_iter_values((0..batch_len).map(|j| {
        let i = batch_start + j;
        let noise = ((i * 7 + 13) % 1000) as f64 / 10_000.0;
        98.6 + noise + rng.random::<f64>() * 1e-10
    }));

    // price: slowly drifting doubles around 1000.
    let price = Float64Array::from_iter_values((0..batch_len).map(|j| {
        let i = batch_start + j;
        1000.0 + (i as f64) * 0.001 + rng.random::<f64>() * 1e-10
    }));

    // measurement: oscillating real doubles.
    let measurement = Float64Array::from_iter_values((0..batch_len).map(|j| {
        let i = batch_start + j;
        (i as f64 * 0.001).sin() * 100.0 + rng.random::<f64>() * 1e-10
    }));

    // temperature: cycling through a few base values with sub-millidegree noise.
    let base_temps = [20.0, 21.5, 22.3, 23.1];
    let temperature = Float64Array::from_iter_values((0..batch_len).map(|j| {
        let i = batch_start + j;
        let base = base_temps[i % base_temps.len()];
        base + ((i * 3 % 997) as f64) * 1e-6 + rng.random::<f64>() * 1e-12
    }));

    // velocity: clustered around 300.0 with gaussian-like noise via Box-Muller.
    let velocity = Float64Array::from_iter_values((0..batch_len).map(|_| {
        let u1: f64 = rng.random::<f64>().max(1e-15);
        let u2: f64 = rng.random::<f64>();
        let normal: f64 = (-2.0_f64 * u1.ln()).sqrt() * (2.0_f64 * std::f64::consts::PI * u2).cos();
        300.0 + normal * 2.0
    }));

    let labels = Int64Array::from_iter_values((0..batch_len).map(|j| {
        let i = batch_start + j;
        (i % 100) as i64
    }));

    RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(ids),
            Arc::new(sensor_reading),
            Arc::new(price),
            Arc::new(measurement),
            Arc::new(temperature),
            Arc::new(velocity),
            Arc::new(labels),
        ],
    )
    .expect("valid record batch")
}
