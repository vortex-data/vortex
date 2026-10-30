// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::borrow::Cow;
use std::fmt;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
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
use object_store::ObjectStore;
use object_store::ObjectStoreExt;
use object_store::path::Path as ObjectPath;
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
use crate::random_access::RemoteDataDir;

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
    /// Immutable directory retained by an explicit remote-fixture cache.
    CachedDir(PathBuf),
    /// Object-store output used by an end-to-end read/decode benchmark.
    Remote(RemoteCompressed),
}

/// Compressed fixture uploaded before timing; reads happen inside each timed iteration.
pub struct RemoteCompressed {
    /// Client shared across iterations; decoded bytes are never cached here.
    pub store: Arc<dyn ObjectStore>,
    /// File key, or the root of a directory-format dataset.
    pub key: ObjectPath,
    /// Fully qualified URI, used by formats with their own native remote reader.
    pub uri: String,
    /// Whether this fixture is a directory-format dataset.
    pub directory: bool,
}

impl Compressed {
    /// Reads an immutable fixture cache shared by real and model runs.
    pub fn from_cache(path: &Path) -> Result<Option<Self>> {
        if !path.exists() {
            return Ok(None);
        }
        let manifest: serde_json::Value =
            serde_json::from_reader(File::open(path.join("manifest.json"))?)?;
        anyhow::ensure!(manifest["version"] == 1, "Unsupported fixture cache");
        let size = manifest["size"]
            .as_u64()
            .context("Missing fixture byte size")?;
        let data = if manifest["directory"]
            .as_bool()
            .context("Missing fixture kind")?
        {
            CompressedData::CachedDir(path.join("data"))
        } else {
            CompressedData::Bytes(Bytes::from(std::fs::read(path.join("data"))?))
        };
        Ok(Some(Self {
            data,
            size,
            elapsed: Duration::ZERO,
        }))
    }

    /// Publishes a complete fixture atomically; failed preparation never leaves a reusable cache.
    pub fn write_cache(&self, path: &Path) -> Result<()> {
        let parent = path
            .parent()
            .context("Fixture cache needs a parent directory")?;
        std::fs::create_dir_all(parent)?;
        let temporary = tempfile::tempdir_in(parent)?;
        let target = temporary.path().join("data");
        let directory = match &self.data {
            CompressedData::Bytes(bytes) => {
                std::fs::write(&target, bytes)?;
                false
            }
            CompressedData::File(file) => {
                std::fs::copy(file.path(), &target)?;
                false
            }
            CompressedData::Dir(_) | CompressedData::CachedDir(_) => {
                let source = match &self.data {
                    CompressedData::Dir(root) => root.path(),
                    CompressedData::CachedDir(root) => root.as_path(),
                    _ => unreachable!(),
                };
                std::fs::create_dir_all(&target)?;
                let mut directories = vec![source.to_path_buf()];
                while let Some(directory) = directories.pop() {
                    for entry in std::fs::read_dir(&directory)? {
                        let entry = entry?;
                        let destination = target.join(entry.path().strip_prefix(source)?);
                        if entry.file_type()?.is_dir() {
                            std::fs::create_dir_all(&destination)?;
                            directories.push(entry.path());
                        } else {
                            anyhow::ensure!(
                                entry.file_type()?.is_file(),
                                "Fixture cache requires regular files"
                            );
                            std::fs::create_dir_all(
                                destination.parent().context("Missing parent")?,
                            )?;
                            std::fs::copy(entry.path(), destination)?;
                        }
                    }
                }
                true
            }
            CompressedData::Remote(_) => bail!("Cache the local compressed fixture before upload"),
        };
        std::fs::write(
            temporary.path().join("manifest.json"),
            serde_json::to_vec(&serde_json::json!({
                "version": 1, "size": self.size, "directory": directory,
            }))?,
        )?;
        std::fs::rename(temporary.path(), path)?;
        Ok(())
    }

    /// Uploads a fixture verbatim outside the read/decode timer.
    pub async fn upload(&self, remote: &RemoteDataDir, prefix: &str) -> Result<Self> {
        let base = remote.object_key(prefix);
        let (key, directory) = match &self.data {
            CompressedData::Bytes(bytes) => {
                let key = base.join("data");
                remote.store().put(&key, bytes.clone().into()).await?;
                (key, false)
            }
            CompressedData::File(file) => {
                let key = base.join("data");
                remote
                    .store()
                    .put(&key, tokio::fs::read(file.path()).await?.into())
                    .await?;
                (key, false)
            }
            CompressedData::Dir(_) | CompressedData::CachedDir(_) => {
                let root = match &self.data {
                    CompressedData::Dir(root) => root.path(),
                    CompressedData::CachedDir(root) => root.as_path(),
                    _ => unreachable!(),
                };
                let mut directories = vec![root.to_path_buf()];
                while let Some(directory) = directories.pop() {
                    for entry in std::fs::read_dir(directory)? {
                        let entry = entry?;
                        if entry.file_type()?.is_dir() {
                            directories.push(entry.path());
                        } else {
                            anyhow::ensure!(
                                entry.file_type()?.is_file(),
                                "Compressed fixtures require regular files"
                            );
                            let path = entry.path();
                            let relative = path
                                .strip_prefix(root)?
                                .to_str()
                                .context("Non-UTF-8 fixture path")?;
                            let key = relative.split('/').fold(base.clone(), ObjectPath::join);
                            remote
                                .store()
                                .put(&key, tokio::fs::read(path).await?.into())
                                .await?;
                        }
                    }
                }
                (base.join("dataset"), true)
            }
            CompressedData::Remote(_) => bail!("Compressed fixture has already been uploaded"),
        };
        Ok(Self {
            data: CompressedData::Remote(RemoteCompressed {
                uri: remote.object_uri(&key)?,
                key,
                directory,
                store: Arc::clone(remote.store()),
            }),
            size: self.size,
            elapsed: self.elapsed,
        })
    }
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

    /// Time a full object GET followed by decoding, including read and open costs.
    /// Directory formats override this with their native remote scanner.
    async fn decompress_remote(&self, remote: &RemoteCompressed) -> Result<Duration> {
        anyhow::ensure!(
            !remote.directory,
            "This format needs a native remote directory reader"
        );
        let start = std::time::Instant::now();
        let bytes = remote.store.get(&remote.key).await?.bytes().await?;
        self.decompress(&Compressed {
            size: bytes.len() as u64,
            data: CompressedData::Bytes(bytes),
            elapsed: Duration::ZERO,
        })
        .await?;
        Ok(start.elapsed())
    }
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
        let elapsed = match &compressed.data {
            CompressedData::Remote(remote) => compressor.decompress_remote(remote).await?,
            _ => compressor.decompress(compressed).await?,
        };

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

    use object_store::memory::InMemory;
    use object_store::throttle::ThrottleConfig;
    use object_store::throttle::ThrottledStore;

    use super::*;

    struct BytesDecoder;

    #[test]
    fn fixture_cache_preserves_file_images_and_directory_layouts() -> Result<()> {
        let directory = tempfile::tempdir()?;
        let file_path = directory.path().join("file");
        assert!(Compressed::from_cache(&file_path)?.is_none());
        Compressed {
            data: CompressedData::Bytes(Bytes::from_static(b"encoded bytes")),
            size: 13,
            elapsed: Duration::ZERO,
        }
        .write_cache(&file_path)?;
        let cached = Compressed::from_cache(&file_path)?.context("Missing cached fixture")?;
        let CompressedData::Bytes(bytes) = cached.data else {
            bail!("Wrong cached fixture kind")
        };
        assert_eq!(bytes.as_ref(), b"encoded bytes");

        let source = tempfile::tempdir()?;
        std::fs::create_dir_all(source.path().join("dataset/data"))?;
        std::fs::write(
            source.path().join("dataset/data/uuid.lance"),
            b"encoded Lance file",
        )?;
        let path = directory.path().join("lance");
        Compressed {
            data: CompressedData::Dir(source),
            size: 18,
            elapsed: Duration::ZERO,
        }
        .write_cache(&path)?;
        let cached = Compressed::from_cache(&path)?.context("Missing cached directory")?;
        let CompressedData::CachedDir(root) = cached.data else {
            bail!("Wrong cached fixture kind")
        };
        assert_eq!(
            std::fs::read(root.join("dataset/data/uuid.lance"))?,
            b"encoded Lance file"
        );
        Ok(())
    }

    #[async_trait]
    impl Compressor for BytesDecoder {
        fn format(&self) -> Format {
            Format::ArrowIpc
        }
        async fn load(&self, _: &Path) -> Result<Uncompressed> {
            bail!("unused in read/decode test")
        }
        async fn compress(&self, _: &Uncompressed) -> Result<Compressed> {
            bail!("unused in read/decode test")
        }
        async fn decompress(&self, compressed: &Compressed) -> Result<Duration> {
            let CompressedData::Bytes(bytes) = &compressed.data else {
                bail!("Expected fetched bytes")
            };
            assert_eq!(bytes.as_ref(), b"compressed fixture");
            Ok(Duration::ZERO)
        }
    }

    #[tokio::test]
    async fn remote_decode_includes_a_fresh_object_read_in_every_iteration() -> Result<()> {
        let store = ThrottledStore::new(
            InMemory::new(),
            ThrottleConfig {
                wait_get_per_call: Duration::from_millis(30),
                ..Default::default()
            },
        );
        let key = ObjectPath::from("fixture");
        store
            .put(&key, Bytes::from_static(b"compressed fixture").into())
            .await?;
        let compressed = Compressed {
            size: 18,
            elapsed: Duration::ZERO,
            data: CompressedData::Remote(RemoteCompressed {
                store: Arc::new(store),
                key,
                uri: "memory://fixture".to_string(),
                directory: false,
            }),
        };
        let result = benchmark_decompress(&BytesDecoder, &compressed, 3, "read+decode").await?;
        assert_eq!(result.all_runs.len(), 3);
        assert!(
            result
                .all_runs
                .iter()
                .all(|elapsed| *elapsed >= Duration::from_millis(30))
        );
        Ok(())
    }

    #[test]
    fn calculate_ratios_adds_vortex_lance_metrics() {
        let mut timings = HashMap::new();
        timings.insert(
            (Format::OnDiskVortex, CompressOp::Compress),
            Duration::from_millis(30),
        );
        timings.insert(
            (Format::Lance, CompressOp::Compress),
            Duration::from_millis(10),
        );
        timings.insert(
            (Format::OnDiskVortex, CompressOp::Decompress),
            Duration::from_millis(24),
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

        assert_eq!(
            ratios
                .iter()
                .map(|m| (m.name.as_str(), m.format, m.unit.as_ref(), m.value))
                .collect::<Vec<_>>(),
            [
                ("vortex:lance size/demo", Format::OnDiskVortex, "ratio", 2.0),
                (
                    "vortex:lance ratio compress time/demo",
                    Format::OnDiskVortex,
                    "ratio",
                    3.0
                ),
                (
                    "vortex:lance ratio decompress time/demo",
                    Format::OnDiskVortex,
                    "ratio",
                    4.0
                ),
            ],
        );
    }
}
