// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::cell::LazyCell;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use anyhow::Result;
use bytes::Bytes;
use clap::ValueEnum;
use parquet::basic::{Compression, ZstdLevel};
use serde::Serialize;
use tokio::runtime::Runtime;
use vortex::Array;
use vortex::arrays::ChunkedVTable;
use vortex::utils::aliases::hash_map::HashMap;

use crate::Format;
use crate::bench_run::run;
use crate::compress::chunked_to_vec_record_batch;
use crate::compress::parquet::{parquet_compress_write, parquet_decompress_read};
use crate::compress::vortex::{vortex_compress_write, vortex_decompress_read};
use crate::measurements::{CompressionTimingMeasurement, CustomUnitMeasurement};

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
    // TODO(connor): This doesn't make a lot of sense...
    let buffer = LazyCell::new(|| {
        let mut buf = Vec::new();
        runtime
            .block_on(vortex_compress_write(uncompressed, &mut buf))
            .expect("Failed to compress with vortex for decompression test");
        Bytes::from(buf)
    });
    // Force materialization of the lazy cell so it's not invoked from within the async benchmark
    // function
    LazyCell::force(&buffer);

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
    let buffer = LazyCell::new(|| {
        let chunked = uncompressed.as_::<ChunkedVTable>().clone();
        let (batches, schema) = chunked_to_vec_record_batch(chunked);
        let mut buf = Vec::new();
        parquet_compress_write(
            batches,
            schema,
            Compression::ZSTD(ZstdLevel::default()),
            &mut buf,
        );
        Bytes::from(buf)
    });
    LazyCell::force(&buffer);

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

pub fn benchmark_lance_compress(
    _runtime: &Runtime,
    _uncompressed: &dyn Array,
    _iterations: usize,
    _bench_name: &str,
) -> Result<(
    Duration,
    u64,
    Vec<CustomUnitMeasurement>,
    CompressionTimingMeasurement,
)> {
    todo!()
}

pub fn benchmark_lance_decompress(
    _runtime: &Runtime,
    _uncompressed: &dyn Array,
    _iterations: usize,
    _bench_name: &str,
) -> Result<(Duration, CompressionTimingMeasurement)> {
    todo!()
}

// Helper function to calculate ratios between formats.
pub fn calculate_ratios(
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

    // Compress time ratio.
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

    // Decompress time ratio.
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
