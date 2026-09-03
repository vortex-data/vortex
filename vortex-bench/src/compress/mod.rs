// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::borrow::Cow;
use std::fmt;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use arrow_array::RecordBatch;
use arrow_schema::Schema;
use async_trait::async_trait;
use bytes::Bytes;
use clap::ValueEnum;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;
use tempfile::NamedTempFile;
use tempfile::TempDir;
use vortex::array::ArrayRef;
use vortex::expr::stats::Stat;
use vortex::utils::aliases::hash_map::HashMap;

use crate::Format;
use crate::measurements::CompressionTimingMeasurement;
use crate::measurements::CustomUnitMeasurement;

/// Number of top-level columns in the wide-table decompression projection benchmark.
pub const READ_PROJECTION_ROOT_COLUMNS: usize = 100_000;

/// Number of top-level columns read by the wide-table decompression projection benchmark.
pub const READ_PROJECTION_COLUMNS: usize = 10_000;

/// Fixed read projection for the wide-table decompression projection benchmark.
pub static READ_PROJECTION: [usize; READ_PROJECTION_COLUMNS] = make_read_projection();

const fn make_read_projection() -> [usize; READ_PROJECTION_COLUMNS] {
    let stride = READ_PROJECTION_ROOT_COLUMNS / READ_PROJECTION_COLUMNS;
    let mut projection = [0; READ_PROJECTION_COLUMNS];
    let mut idx = 0;
    while idx < READ_PROJECTION_COLUMNS {
        projection[idx] = idx * stride;
        idx += 1;
    }
    projection
}

/// Read projection for a file with `root_columns` top-level columns, if this benchmark projects it.
pub fn read_projection(root_columns: usize) -> Option<&'static [usize]> {
    (root_columns == READ_PROJECTION_ROOT_COLUMNS).then_some(&READ_PROJECTION)
}

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

/// Source data in the form a format compresses, read from Parquet once per format.
pub enum Uncompressed {
    /// Arrow record batches sharing one schema.
    Arrow {
        schema: Arc<Schema>,
        batches: Vec<RecordBatch>,
    },
    /// A Vortex array.
    Vortex(ArrayRef),
    /// Format-private input, for backends whose Arrow crate version differs from the
    /// workspace's and so cannot share [`Self::Arrow`].
    Opaque(Box<dyn Any + Send + Sync>),
}

impl Uncompressed {
    /// Read a Parquet file into Arrow record batches.
    pub fn read_arrow(parquet_path: &Path) -> Result<Self> {
        let builder = ParquetRecordBatchReaderBuilder::try_new(File::open(parquet_path)?)?;
        let schema = Arc::clone(builder.schema());
        let batches = builder.build()?.collect::<Result<Vec<_>, _>>()?;

        Ok(Self::Arrow { schema, batches })
    }

    /// The Arrow schema and batches, or an error if this is not Arrow data.
    pub fn arrow(&self) -> Result<(&Arc<Schema>, &[RecordBatch])> {
        match self {
            Self::Arrow { schema, batches } => Ok((schema, batches)),
            _ => bail!("expected Arrow record batches"),
        }
    }

    /// The Vortex array, or an error if this is not Vortex data.
    pub fn vortex(&self) -> Result<&ArrayRef> {
        match self {
            Self::Vortex(array) => Ok(array),
            _ => bail!("expected a Vortex array"),
        }
    }

    /// The format-private input as `T`, or an error if this holds something else.
    pub fn opaque<T: Any>(&self) -> Result<&T> {
        match self {
            Self::Opaque(input) => input
                .downcast_ref()
                .ok_or_else(|| anyhow::anyhow!("format-private input has an unexpected type")),
            _ => bail!("expected format-private input"),
        }
    }

    /// Return the input to the state [`Self::read_arrow`] or a fresh conversion would produce.
    ///
    /// The Vortex writer computes statistics inside the timed region and caches them on the
    /// array, so reusing one array across iterations would let every run after the first skip
    /// that work. Clearing the cache keeps each iteration's measurement comparable.
    pub fn reset(&self) {
        if let Self::Vortex(array) = self {
            clear_stats(array);
        }
    }
}

/// Clear cached statistics on `array` and every array beneath it.
fn clear_stats(array: &ArrayRef) {
    for stat in Stat::all() {
        array.statistics().clear(stat);
    }
    for child in array.children_iter() {
        clear_stats(child);
    }
}

/// Where a format keeps its compressed output.
///
/// Temporary files and directories are removed on drop, so the output lives exactly as long as
/// the value does.
pub enum CompressedData {
    /// In-memory file image.
    Bytes(Bytes),
    /// Single file on disk.
    File(NamedTempFile),
    /// Directory of files on disk.
    Dir(TempDir),
}

/// Output of one compression run: the compressed data, its size and the compression time.
pub struct Compressed {
    pub data: CompressedData,
    pub size: u64,
    pub elapsed: Duration,
}

/// Result of a compression benchmark run.
pub struct CompressResult {
    pub time: Duration,
    pub compressed_size: u64,
    /// Output of the last iteration, kept so decompression need not compress again.
    pub compressed: Compressed,
    pub timing: CompressionTimingMeasurement,
    /// Per-iteration encode wall times. Captured for v3 emission.
    pub all_runs: Vec<Duration>,
    pub ratios: Vec<CustomUnitMeasurement>,
}

/// Result of a decompression benchmark run.
pub struct DecompressResult {
    pub time: Duration,
    pub timing: CompressionTimingMeasurement,
    /// Per-iteration decode wall times. Captured for v3 emission.
    pub all_runs: Vec<Duration>,
}

/// Trait for format-specific compression/decompression operations.
///
/// Implementations handle the actual compression logic for a specific format
/// (e.g., Vortex, Parquet, Lance). The benchmark functions use this trait
/// to run timing measurements.
///
/// The input data is provided as a path to a Parquet file. [`Self::load`] reads it once into
/// the form the format compresses; the timed operations never touch the file again.
#[async_trait]
pub trait Compressor: Send + Sync {
    /// The format this compressor handles.
    fn format(&self) -> Format;

    /// Read a Parquet file into the form [`Self::compress`] consumes. Not timed.
    async fn load(&self, parquet_path: &Path) -> Result<Uncompressed>;

    /// Compress output previously produced by [`Self::load`].
    ///
    /// Only the compression itself should be timed.
    async fn compress(&self, input: &Uncompressed) -> Result<Compressed>;

    /// Decompress output previously produced by [`Self::compress`].
    ///
    /// The timing returned should only measure the decompression phase.
    ///
    /// Format implementations apply the fixed wide-table read projection when the input schema
    /// matches the projection benchmark.
    async fn decompress(&self, compressed: &Compressed) -> Result<Duration>;
}

/// Run a compression benchmark for the given compressor.
///
/// Compresses the same `input` `iterations` times and returns timing statistics. The input is
/// [reset](Uncompressed::reset) before every iteration so none of them starts warm.
pub async fn benchmark_compress(
    compressor: &dyn Compressor,
    input: &Uncompressed,
    iterations: usize,
    bench_name: &str,
) -> Result<CompressResult> {
    let format = compressor.format();
    let mut fastest = Duration::MAX;
    let mut all_runs = Vec::with_capacity(iterations);
    let mut compressed = None;

    for _ in 0..iterations {
        input.reset();
        let result = compressor.compress(input).await?;

        fastest = fastest.min(result.elapsed);
        all_runs.push(result.elapsed);
        compressed = Some(result);
    }

    let compressed = compressed.context("--iterations must be at least 1")?;
    let compressed_size = compressed.size;

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
        compressed,
        timing,
        all_runs,
        ratios,
    })
}

/// Run a decompression benchmark for the given compressor.
///
/// Decompresses the same `compressed` output `iterations` times.
pub async fn benchmark_decompress(
    compressor: &dyn Compressor,
    compressed: &Compressed,
    iterations: usize,
    bench_name: &str,
) -> Result<DecompressResult> {
    let format = compressor.format();
    let mut fastest = Duration::MAX;
    let mut all_runs = Vec::with_capacity(iterations);

    for _ in 0..iterations {
        let elapsed = compressor.decompress(compressed).await?;

        fastest = fastest.min(elapsed);
        all_runs.push(elapsed);
    }

    let timing = CompressionTimingMeasurement {
        name: format!("decompress time/{bench_name}"),
        time: fastest,
        format,
    };

    Ok(DecompressResult {
        time: fastest,
        timing,
        all_runs,
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
