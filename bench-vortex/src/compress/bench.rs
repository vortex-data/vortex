// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;
use std::time::Duration;

use anyhow::Result;
use bytes::Bytes;
use clap::ValueEnum;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use serde::Serialize;
use tokio::runtime::Runtime;
use vortex::array::Array;
use vortex::array::arrays::ChunkedVTable;
use vortex::utils::aliases::hash_map::HashMap;
#[cfg(feature = "lance")]
#[rustfmt::skip]
use {
    super::lance::*,
    crate::bench_run::run_with_setup,
    crate::utils::convert_utf8view_batch,
    crate::utils::convert_utf8view_schema,
    arrow_array::RecordBatch,
    parking_lot::Mutex,
    std::fs,
    std::path::PathBuf,
    std::sync::Arc,
};

use crate::Format;
use crate::bench_run::run;
use crate::compress::chunked_to_vec_record_batch;
use crate::compress::parquet::parquet_compress_write;
use crate::compress::parquet::parquet_decompress_read;
use crate::compress::vortex::vortex_compress_write;
use crate::compress::vortex::vortex_decompress_read;
use crate::measurements::CompressionTimingMeasurement;
use crate::measurements::CustomUnitMeasurement;

#[derive(Default)]
pub struct CompressMeasurements {
    pub timings: Vec<CompressionTimingMeasurement>,
    pub ratios: Vec<CustomUnitMeasurement>,
}

impl Extend<CompressMeasurements> for CompressMeasurements {
    fn extend<T: IntoIterator<Item = CompressMeasurements>>(&mut self, iter: T) {
        iter.into_iter().for_each(|measurement| {
            self.timings.extend(measurement.timings);
            self.ratios.extend(measurement.ratios);
        })
    }
}

impl FromIterator<CompressMeasurements> for CompressMeasurements {
    fn from_iter<T: IntoIterator<Item = CompressMeasurements>>(iter: T) -> Self {
        let mut into_iter = iter.into_iter();
        match into_iter.next() {
            None => CompressMeasurements::default(),
            Some(mut ms) => {
                ms.extend(into_iter);
                ms
            }
        }
    }
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq, ValueEnum, Serialize)]
pub enum CompressOp {
    Compress,
    Decompress,
}

impl fmt::Display for CompressOp {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            CompressOp::Compress => write!(f, "Compress"),
            CompressOp::Decompress => write!(f, "Decompress"),
        }
    }
}

pub fn benchmark_vortex_compress(
    runtime: &Runtime,
    uncompressed: &dyn Array,
    iterations: usize,
    bench_name: &str,
) -> Result<(
    Duration,
    u64,
    Vec<CustomUnitMeasurement>,
    CompressionTimingMeasurement,
)> {
    let compressed_size = AtomicU64::default();

    // Run the benchmark and measure time.
    let time = run(runtime, iterations, || async {
        compressed_size.store(
            vortex_compress_write(uncompressed, &mut Vec::new())
                .await
                .expect("Failed to compress with vortex"),
            Ordering::SeqCst,
        );
    });

    let compressed_size_val = compressed_size.load(Ordering::SeqCst);
    let ratios = vec![CustomUnitMeasurement {
        name: format!("vortex size/{bench_name}"),
        format: Format::OnDiskVortex,
        unit: Cow::from("bytes"),
        value: compressed_size_val as f64,
    }];

    let timing = CompressionTimingMeasurement {
        name: format!("compress time/{bench_name}"),
        time,
        format: Format::OnDiskVortex,
    };

    Ok((time, compressed_size_val, ratios, timing))
}

pub fn benchmark_vortex_decompress(
    runtime: &Runtime,
    uncompressed: &dyn Array,
    iterations: usize,
    bench_name: &str,
) -> Result<(Duration, CompressionTimingMeasurement)> {
    let mut buf = Vec::new();
    runtime
        .block_on(vortex_compress_write(uncompressed, &mut buf))
        .expect("Failed to compress with vortex for decompression test");
    let buffer = Bytes::from(buf);

    // Run the benchmark and measure time.
    let time = run(runtime, iterations, || async {
        vortex_decompress_read(buffer.clone())
            .await
            .expect("Failed to decompress with vortex")
    });

    let timing = CompressionTimingMeasurement {
        name: format!("decompress time/{bench_name}"),
        time,
        format: Format::OnDiskVortex,
    };

    Ok((time, timing))
}

pub fn benchmark_parquet_compress(
    runtime: &Runtime,
    uncompressed: &dyn Array,
    iterations: usize,
    bench_name: &str,
) -> Result<(
    Duration,
    u64,
    Vec<CustomUnitMeasurement>,
    CompressionTimingMeasurement,
)> {
    let parquet_compressed_size = AtomicU64::default();
    let chunked = uncompressed.as_::<ChunkedVTable>().clone();
    let (batches, schema) = chunked_to_vec_record_batch(chunked);

    // Run the benchmark and measure time.
    let time = run(runtime, iterations, || async {
        parquet_compressed_size.store(
            parquet_compress_write(
                batches.clone(),
                schema.clone(),
                Compression::ZSTD(ZstdLevel::default()),
                &mut Vec::new(),
            ) as u64,
            Ordering::SeqCst,
        );
    });

    let parquet_compressed_size_val = parquet_compressed_size.into_inner();
    let ratios = vec![CustomUnitMeasurement {
        name: format!("parquet-zstd size/{bench_name}"),
        // unlike timings, ratios have a single column vortex
        format: Format::OnDiskVortex,
        unit: Cow::from("bytes"),
        value: parquet_compressed_size_val as f64,
    }];

    let timing = CompressionTimingMeasurement {
        name: format!("compress time/{bench_name}"),
        time,
        format: Format::Parquet,
    };

    Ok((time, parquet_compressed_size_val, ratios, timing))
}

pub fn benchmark_parquet_decompress(
    runtime: &Runtime,
    uncompressed: &dyn Array,
    iterations: usize,
    bench_name: &str,
) -> Result<(Duration, CompressionTimingMeasurement)> {
    let chunked = uncompressed.as_::<ChunkedVTable>().clone();
    let (batches, schema) = chunked_to_vec_record_batch(chunked);
    let mut buf = Vec::new();
    parquet_compress_write(
        batches,
        schema,
        Compression::ZSTD(ZstdLevel::default()),
        &mut buf,
    );
    let buffer = Bytes::from(buf);

    // Run the benchmark and measure time.
    let time = run(runtime, iterations, || async {
        parquet_decompress_read(buffer.clone());
    });

    let timing = CompressionTimingMeasurement {
        name: format!("decompress time/{bench_name}"),
        time,
        format: Format::Parquet,
    };

    Ok((time, timing))
}

#[cfg(feature = "lance")]
pub fn benchmark_lance_compress(
    runtime: &Runtime,
    uncompressed: &dyn Array,
    iterations: usize,
    bench_name: &str,
) -> Result<(
    Duration,
    u64,
    Vec<CustomUnitMeasurement>,
    CompressionTimingMeasurement,
)> {
    // NOTE: Lance requires filesystem access unlike Parquet/Vortex which use in-memory buffers.
    // To make the benchmark fairer, we exclude directory creation and size calculation from timing
    // (which is included in timing in the other benchmarks).

    let chunked = uncompressed.as_::<ChunkedVTable>().clone();
    let (batches, schema) = chunked_to_vec_record_batch(chunked);

    // Convert Utf8View to Utf8 (Lance doesn't support Utf8View).
    let converted_batches: Vec<RecordBatch> = batches
        .into_iter()
        .map(convert_utf8view_batch)
        .collect::<Result<Vec<_>, _>>()?;
    let converted_schema = convert_utf8view_schema(&schema);

    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");
    let iteration_paths: Arc<Mutex<Vec<PathBuf>>> = Arc::new(Mutex::new(Vec::new()));
    let iteration_counter = AtomicU64::new(0);

    // Run the benchmark and measure time.
    let time = run_with_setup(
        runtime,
        iterations,
        || {
            // Create a unique subdirectory for each iteration (not timed).
            let iteration_id = iteration_counter.fetch_add(1, Ordering::Relaxed);
            let iteration_dir = temp_dir.path().join(format!("iter_{}", iteration_id));
            fs::create_dir_all(&iteration_dir).expect("Failed to create iteration directory");

            (
                iteration_dir,
                converted_batches.clone(),
                converted_schema.clone(),
                iteration_paths.clone(),
            )
        },
        |(iteration_dir, batches, schema, paths)| async move {
            lance_compress_write_only(batches, schema, &iteration_dir)
                .await
                .expect("Failed to compress with lance");

            // Since there should be low contention, this won't block and will be fast.
            paths.lock().push(iteration_dir);
        },
    );

    // Calculate size from the last iteration.
    let paths = iteration_paths.lock();
    let lance_compressed_size_val = if let Some(last_path) = paths.last() {
        calculate_lance_size(last_path).expect("Failed to calculate Lance size")
    } else {
        0
    };
    let ratios = vec![CustomUnitMeasurement {
        name: format!("lance size/{bench_name}"),
        // Unlike timings, ratios have a single column vortex.
        format: Format::OnDiskVortex,
        unit: Cow::from("bytes"),
        value: lance_compressed_size_val as f64,
    }];

    let timing = CompressionTimingMeasurement {
        name: format!("compress time/{bench_name}"),
        time,
        format: Format::Lance,
    };

    Ok((time, lance_compressed_size_val, ratios, timing))
}

#[cfg(feature = "lance")]
pub fn benchmark_lance_decompress(
    runtime: &Runtime,
    uncompressed: &dyn Array,
    iterations: usize,
    bench_name: &str,
) -> Result<(Duration, CompressionTimingMeasurement)> {
    // NOTE: Lance requires filesystem access unlike Parquet/Vortex which use in-memory buffers.
    let chunked = uncompressed.as_::<ChunkedVTable>().clone();
    let (batches, schema) = chunked_to_vec_record_batch(chunked);
    let temp_dir = tempfile::tempdir().expect("Failed to create temp dir");

    // Write the Lance dataset once for all iterations.
    let dataset_path = runtime.block_on(async {
        lance_compress_write(batches, schema, &temp_dir)
            .await
            .expect("Failed to compress with lance for decompression test")
    });

    // Keep temp_dir alive to prevent deletion.
    let temp_path = (dataset_path, temp_dir);

    // Run the benchmark and measure time.
    let time = run(runtime, iterations, || async {
        lance_decompress_read(&temp_path.0)
            .await
            .expect("Failed to decompress with lance");
    });

    let timing = CompressionTimingMeasurement {
        name: format!("decompress time/{bench_name}"),
        time,
        format: Format::Lance,
    };

    Ok((time, timing))
}

// Helper function to calculate ratios between formats.
pub fn calculate_ratios(
    measurements: &HashMap<(Format, CompressOp), Duration>,
    compressed_sizes: &HashMap<Format, u64>,
    bench_name: &str,
    ratios: &mut Vec<CustomUnitMeasurement>,
) {
    calculate_vortex_parquet_ratios(measurements, compressed_sizes, bench_name, ratios);

    #[cfg(feature = "lance")]
    calculate_vortex_lance_ratios(measurements, compressed_sizes, bench_name, ratios);
}

fn calculate_vortex_parquet_ratios(
    measurements: &HashMap<(Format, CompressOp), Duration>,
    compressed_sizes: &HashMap<Format, u64>,
    bench_name: &str,
    ratios: &mut Vec<CustomUnitMeasurement>,
) {
    // Size ratio: vortex vs parquet.
    if let (Some(vortex_size), Some(parquet_size)) = (
        compressed_sizes.get(&Format::OnDiskVortex),
        compressed_sizes.get(&Format::Parquet),
    ) {
        ratios.push(CustomUnitMeasurement {
            name: format!("vortex:parquet-zstd size/{bench_name}"),
            format: Format::OnDiskVortex,
            unit: Cow::from("ratio"),
            value: *vortex_size as f64 / *parquet_size as f64,
        });
    }

    // Compress time ratio: vortex vs parquet.
    if let (Some(vortex_time), Some(parquet_time)) = (
        measurements.get(&(Format::OnDiskVortex, CompressOp::Compress)),
        measurements.get(&(Format::Parquet, CompressOp::Compress)),
    ) {
        ratios.push(CustomUnitMeasurement {
            name: format!("vortex:parquet-zstd ratio compress time/{bench_name}"),
            format: Format::OnDiskVortex,
            unit: Cow::from("ratio"),
            value: vortex_time.as_nanos() as f64 / parquet_time.as_nanos() as f64,
        });
    }

    // Decompress time ratio: vortex vs parquet.
    if let (Some(vortex_time), Some(parquet_time)) = (
        measurements.get(&(Format::OnDiskVortex, CompressOp::Decompress)),
        measurements.get(&(Format::Parquet, CompressOp::Decompress)),
    ) {
        ratios.push(CustomUnitMeasurement {
            name: format!("vortex:parquet-zstd ratio decompress time/{bench_name}"),
            format: Format::OnDiskVortex,
            unit: Cow::from("ratio"),
            value: vortex_time.as_nanos() as f64 / parquet_time.as_nanos() as f64,
        });
    }
}

#[cfg(feature = "lance")]
fn calculate_vortex_lance_ratios(
    measurements: &HashMap<(Format, CompressOp), Duration>,
    compressed_sizes: &HashMap<Format, u64>,
    bench_name: &str,
    ratios: &mut Vec<CustomUnitMeasurement>,
) {
    // Size ratio: vortex vs lance.
    if let (Some(vortex_size), Some(lance_size)) = (
        compressed_sizes.get(&Format::OnDiskVortex),
        compressed_sizes.get(&Format::Lance),
    ) {
        ratios.push(CustomUnitMeasurement {
            name: format!("vortex:lance size/{bench_name}"),
            format: Format::OnDiskVortex,
            unit: Cow::from("ratio"),
            value: *vortex_size as f64 / *lance_size as f64,
        });
    }

    // Compress time ratio: vortex vs lance.
    if let (Some(vortex_time), Some(lance_time)) = (
        measurements.get(&(Format::OnDiskVortex, CompressOp::Compress)),
        measurements.get(&(Format::Lance, CompressOp::Compress)),
    ) {
        ratios.push(CustomUnitMeasurement {
            name: format!("vortex:lance ratio compress time/{bench_name}"),
            format: Format::OnDiskVortex,
            unit: Cow::from("ratio"),
            value: vortex_time.as_nanos() as f64 / lance_time.as_nanos() as f64,
        });
    }

    // Decompress time ratio: vortex vs lance.
    if let (Some(vortex_time), Some(lance_time)) = (
        measurements.get(&(Format::OnDiskVortex, CompressOp::Decompress)),
        measurements.get(&(Format::Lance, CompressOp::Decompress)),
    ) {
        ratios.push(CustomUnitMeasurement {
            name: format!("vortex:lance ratio decompress time/{bench_name}"),
            format: Format::OnDiskVortex,
            unit: Cow::from("ratio"),
            value: vortex_time.as_nanos() as f64 / lance_time.as_nanos() as f64,
        });
    }
}
