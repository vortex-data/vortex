// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::borrow::Cow;
use std::fmt;
use std::path::Path;
use std::time::Duration;

use anyhow::Result;
use async_trait::async_trait;
use clap::ValueEnum;
use serde::Serialize;
use vortex::utils::aliases::hash_map::HashMap;

use crate::Format;
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

/// Result of a compression benchmark run.
pub struct CompressResult {
    pub time: Duration,
    pub compressed_size: u64,
    pub timing: CompressionTimingMeasurement,
    pub ratios: Vec<CustomUnitMeasurement>,
}

/// Result of a decompression benchmark run.
pub struct DecompressResult {
    pub time: Duration,
    pub timing: CompressionTimingMeasurement,
}

/// Trait for format-specific compression/decompression operations.
///
/// Implementations handle the actual compression logic for a specific format
/// (e.g., Vortex, Parquet, Lance). The benchmark functions use this trait
/// to run timing measurements.
///
/// The input data is provided as a path to a Parquet file, which implementations
/// read and convert as needed for their target format.
#[async_trait]
pub trait Compressor: Send + Sync {
    /// The format this compressor handles.
    fn format(&self) -> Format;

    /// Compress data from a Parquet file, returning the compressed size in bytes and elapsed time.
    ///
    /// The implementation should read the Parquet file and compress it
    /// to the target format.
    async fn compress(&self, parquet_path: &Path) -> Result<(u64, Duration)>;

    /// Decompress data from the Parquet file (after compressing), returning the decompressed size.
    ///
    /// This method first compresses the data to the target format, then decompresses it.
    /// The timing returned should only measure the decompression phase.
    async fn decompress(&self, parquet_path: &Path) -> Result<Duration>;
}

/// Run a compression benchmark for the given compressor.
///
/// Executes compression `iterations` times and returns timing statistics.
pub async fn benchmark_compress(
    compressor: &dyn Compressor,
    parquet_path: &Path,
    iterations: usize,
    bench_name: &str,
) -> Result<CompressResult> {
    let format = compressor.format();
    let mut fastest = Duration::MAX;
    let mut compressed_size = 0u64;

    for _ in 0..iterations {
        let (size, elapsed) = compressor.compress(parquet_path).await?;

        compressed_size = size;
        fastest = fastest.min(elapsed);
    }

    let ratios = vec![CustomUnitMeasurement {
        name: format!("{} size/{bench_name}", format.name()),
        format,
        unit: Cow::from("bytes"),
        value: compressed_size as f64,
    }];

    let timing = CompressionTimingMeasurement {
        name: format!("compress time/{bench_name}"),
        time: fastest,
        format,
    };

    Ok(CompressResult {
        time: fastest,
        compressed_size,
        timing,
        ratios,
    })
}

/// Run a decompression benchmark for the given compressor.
///
/// Benchmarks decompression `iterations` times.
pub async fn benchmark_decompress(
    compressor: &dyn Compressor,
    parquet_path: &Path,
    iterations: usize,
    bench_name: &str,
) -> Result<DecompressResult> {
    let format = compressor.format();
    let mut fastest = Duration::MAX;

    for _ in 0..iterations {
        let elapsed = compressor.decompress(parquet_path).await?;

        fastest = fastest.min(elapsed);
    }

    let timing = CompressionTimingMeasurement {
        name: format!("decompress time/{bench_name}"),
        time: fastest,
        format,
    };

    Ok(DecompressResult {
        time: fastest,
        timing,
    })
}

/// Calculate cross-format comparison ratios.
pub fn calculate_ratios(
    measurements: &HashMap<(Format, CompressOp), Duration>,
    compressed_sizes: &HashMap<Format, u64>,
    bench_name: &str,
    ratios: &mut Vec<CustomUnitMeasurement>,
) {
    calculate_vortex_parquet_ratios(measurements, compressed_sizes, bench_name, ratios);
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

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[test]
    fn calculate_ratios_adds_vortex_lance_metrics() {
        let mut timings = HashMap::new();
        timings.insert(
            (Format::OnDiskVortex, CompressOp::Compress),
            Duration::from_millis(20),
        );
        timings.insert(
            (Format::Lance, CompressOp::Compress),
            Duration::from_millis(10),
        );
        timings.insert(
            (Format::OnDiskVortex, CompressOp::Decompress),
            Duration::from_millis(12),
        );
        timings.insert(
            (Format::Lance, CompressOp::Decompress),
            Duration::from_millis(6),
        );

        let mut compressed_sizes = HashMap::new();
        compressed_sizes.insert(Format::OnDiskVortex, 400);
        compressed_sizes.insert(Format::Lance, 200);

        let mut ratios = Vec::new();
        calculate_ratios(&timings, &compressed_sizes, "demo", &mut ratios);

        assert!(
            ratios
                .iter()
                .any(|m| m.name == "vortex:lance size/demo" && m.value == 2.0)
        );
        assert!(
            ratios
                .iter()
                .any(|m| { m.name == "vortex:lance ratio compress time/demo" && m.value == 2.0 })
        );
        assert!(
            ratios
                .iter()
                .any(|m| { m.name == "vortex:lance ratio decompress time/demo" && m.value == 2.0 })
        );
    }
}
