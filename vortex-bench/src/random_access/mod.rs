// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use anyhow::anyhow;
use arrow_array::RecordBatch;
use arrow_ipc::writer::FileWriter;
use async_trait::async_trait;
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;
use url::Url;
use vortex::array::ArrayRef;

use crate::Format;
use crate::data_dir;
use crate::idempotent;

pub mod take;

// Re-export implementations
pub use take::ArrowIpcRandomAccessor;
pub use take::ParquetRandomAccessor;
pub use take::VortexRandomAccessor;

pub const ARROW_ROW_OFFSETS_METADATA_KEY: &str = "vortex.row_offsets";

/// Generate the data path for a random-access benchmark dataset file.
///
/// Returns a path like `random_access/{dataset}/{dataset}.{ext}`
/// (or `{dataset}-compact.{ext}` for [`Format::VortexCompact`]).
pub fn data_path(dataset: &str, format: Format) -> String {
    let ext = format.ext();
    match format {
        Format::VortexCompact => format!("random_access/{dataset}/{dataset}-compact.{ext}"),
        _ => format!("random_access/{dataset}/{dataset}.{ext}"),
    }
}

/// Convert a Parquet input into an uncompressed Arrow IPC file.
pub fn parquet_to_arrow_file(parquet_path: PathBuf, arrow_path: String) -> Result<PathBuf> {
    idempotent(&arrow_path, |output_path| {
        let input = File::open(parquet_path)?;
        let builder = ParquetRecordBatchReaderBuilder::try_new(input)?;
        let schema = Arc::clone(builder.schema());
        let reader = builder.build()?;
        let output = File::create(output_path)?;
        let mut writer = FileWriter::try_new(output, schema.as_ref())?;
        let mut row_offsets = vec![0_u64];

        for batch in reader {
            let batch = batch?;
            let next_offset = row_offsets
                .last()
                .copied()
                .unwrap_or_default()
                .checked_add(u64::try_from(batch.num_rows())?)
                .ok_or_else(|| anyhow::anyhow!("Arrow row count overflow"))?;
            writer.write(&batch)?;
            row_offsets.push(next_offset);
        }

        writer.write_metadata(
            ARROW_ROW_OFFSETS_METADATA_KEY,
            row_offsets
                .iter()
                .map(u64::to_string)
                .collect::<Vec<_>>()
                .join(","),
        );
        writer.finish()?;
        Ok(())
    })
}

/// Approximate byte budget for one row group in the synthetic random-access datasets.
const TARGET_ROW_GROUP_BYTES: usize = 128 * 1024 * 1024;

/// Row groups are sized in whole multiples of this many rows.
///
/// Parquet is converted to Vortex by streaming Arrow batches of [`PARQUET_READ_BATCH_SIZE`] rows,
/// so keeping row groups a whole multiple of it leaves the derived Vortex files byte-identical:
/// only the Parquet layout changes.
const PARQUET_READ_BATCH_SIZE: usize = 1024;

/// Bounds on the row group size, in units of [`PARQUET_READ_BATCH_SIZE`] rows.
///
/// The upper bound matters more than the byte budget for narrow rows: without it a million-row
/// dataset lands in a single row group, and a point lookup then has to read the entire file.
const MIN_BATCHES_PER_ROW_GROUP: usize = 8;
const MAX_BATCHES_PER_ROW_GROUP: usize = 64;

/// Rows per data page.
///
/// Finer pages than the 20k-row default give the page index enough resolution to be useful for
/// point lookups, at the cost of a slightly larger index.
const DATA_PAGE_ROWS: usize = PARQUET_READ_BATCH_SIZE;

/// Parquet writer properties for a synthetic random-access dataset of `approx_row_bytes` per row.
///
/// The defaults are wrong for this suite: `max_row_group_size` defaults to 1Mi rows, which is more
/// than every dataset here holds, so each file ends up as one row group. Readers select row groups
/// before rows, so a single-row-group file forces a point lookup to fetch and decode the whole
/// file — cheap from page cache, ruinous over an object store.
pub fn random_access_writer_properties(approx_row_bytes: usize) -> WriterProperties {
    let batches = (TARGET_ROW_GROUP_BYTES / approx_row_bytes / PARQUET_READ_BATCH_SIZE)
        .clamp(MIN_BATCHES_PER_ROW_GROUP, MAX_BATCHES_PER_ROW_GROUP);

    WriterProperties::builder()
        .set_max_row_group_size(batches * PARQUET_READ_BATCH_SIZE)
        .set_data_page_row_count_limit(DATA_PAGE_ROWS)
        .build()
}

/// A remote directory holding the same layout as the local benchmark data directory.
///
/// Random access datasets are always materialized locally first, then uploaded verbatim, so a
/// remote object key is just the local path relative to [`data_dir`] appended to the URL path.
#[derive(Clone, Debug)]
pub struct RemoteDataDir {
    url: Url,
    store: Arc<dyn ObjectStore>,
}

impl RemoteDataDir {
    /// Build an object store for `url` (e.g. `s3://bucket/prefix/`) from the ambient environment.
    pub fn try_new(url: Url) -> Result<Self> {
        let store: Arc<dyn ObjectStore> = match url.scheme() {
            "s3" => {
                let bucket = url
                    .host_str()
                    .ok_or_else(|| anyhow!("remote data dir has no bucket: {url}"))?;
                Arc::new(
                    AmazonS3Builder::from_env()
                        .with_bucket_name(bucket)
                        .build()?,
                )
            }
            other => return Err(anyhow!("unsupported remote data dir scheme: {other}")),
        };
        Ok(Self { url, store })
    }

    /// The object store backing this directory.
    pub fn store(&self) -> &Arc<dyn ObjectStore> {
        &self.store
    }

    /// The object key of `local_path`, mirroring its location under the local data directory.
    pub fn key(&self, local_path: &Path) -> Result<String> {
        let relative = local_path.strip_prefix(data_dir()).map_err(|_| {
            anyhow!(
                "{} is not inside the benchmark data directory",
                local_path.display()
            )
        })?;
        let relative = relative
            .to_str()
            .ok_or_else(|| anyhow!("non-UTF-8 data path: {}", local_path.display()))?;
        let prefix = self
            .url
            .path()
            .trim_start_matches('/')
            .trim_end_matches('/');
        Ok(if prefix.is_empty() {
            relative.to_string()
        } else {
            format!("{prefix}/{relative}")
        })
    }

    /// The fully qualified URL of `local_path` in this remote directory.
    pub fn uri(&self, local_path: &Path) -> Result<String> {
        let scheme = self.url.scheme();
        let host = self
            .url
            .host_str()
            .ok_or_else(|| anyhow!("remote data dir has no bucket: {}", self.url))?;
        Ok(format!("{scheme}://{host}/{}", self.key(local_path)?))
    }
}

/// Trait for a benchmark dataset that knows how to prepare data files.
#[async_trait]
pub trait BenchDataset: Send + Sync {
    /// A descriptive name for this dataset (used in benchmark output and CLI).
    fn name(&self) -> &str;

    /// The total number of rows in this dataset.
    fn row_count(&self) -> u64;

    /// Prepare the data file for the given format and return its path.
    ///
    /// This writes the file if it doesn't already exist.
    async fn path(&self, format: Format) -> Result<PathBuf>;
}

pub enum RandomAccessorRet {
    RecordBatch(RecordBatch),
    ArrayRef(ArrayRef),
    Native(Box<dyn std::any::Any + Send>),
}

/// Trait for format-specific random access (take) operations.
///
/// Implementations handle reading specific rows by index from a data source.
/// Accessors are constructed in a ready-to-use state with metadata already parsed.
#[async_trait]
pub trait RandomAccessor: Send + Sync {
    /// A descriptive name for this accessor (used in benchmark output).
    fn name(&self) -> &str;

    /// The format this accessor handles.
    fn format(&self) -> Format;

    /// Take rows at the given indices, returning the handle.
    async fn take(&self, indices: &[u64]) -> Result<RandomAccessorRet>;
}

#[cfg(test)]
mod tests {
    use rstest::rstest;

    use super::*;

    fn remote(url: &str) -> Result<RemoteDataDir> {
        // `from_env` needs no credentials to construct the client.
        RemoteDataDir::try_new(Url::parse(url)?)
    }

    #[test]
    fn key_mirrors_the_local_data_dir_layout() -> Result<()> {
        let local = data_dir().join("random_access/taxi/taxi.vortex");

        assert_eq!(
            remote("s3://bucket/prefix/")?.key(&local)?,
            "prefix/random_access/taxi/taxi.vortex"
        );
        assert_eq!(
            remote("s3://bucket/")?.key(&local)?,
            "random_access/taxi/taxi.vortex"
        );
        assert_eq!(
            remote("s3://bucket/prefix/")?.uri(&local)?,
            "s3://bucket/prefix/random_access/taxi/taxi.vortex"
        );
        Ok(())
    }

    #[test]
    fn key_rejects_paths_outside_the_data_dir() -> Result<()> {
        assert!(
            remote("s3://bucket/prefix/")?
                .key(Path::new("/tmp/taxi.vortex"))
                .is_err()
        );
        Ok(())
    }

    /// Row widths of the three synthetic random-access datasets.
    const FEATURE_VECTORS_ROW_BYTES: usize = 8 + 1024 * 4;
    const NESTED_LISTS_ROW_BYTES: usize = 8 + 10 * 8;
    const NESTED_STRUCTS_ROW_BYTES: usize = 8 * 6;

    #[rstest]
    #[case::feature_vectors(FEATURE_VECTORS_ROW_BYTES)]
    #[case::nested_lists(NESTED_LISTS_ROW_BYTES)]
    #[case::nested_structs(NESTED_STRUCTS_ROW_BYTES)]
    fn row_groups_split_a_million_row_dataset(#[case] approx_row_bytes: usize) {
        let props = random_access_writer_properties(approx_row_bytes);
        let rows_per_group = props.max_row_group_size();

        // The whole point: a million-row dataset must not land in a single row group.
        assert!(
            rows_per_group < 1_000_000,
            "{approx_row_bytes} byte rows produced one row group of {rows_per_group}"
        );
        // Row group boundaries stay aligned to the Arrow batches the Vortex conversion reads,
        // so the derived Vortex files are unaffected by this layout.
        assert_eq!(rows_per_group % PARQUET_READ_BATCH_SIZE, 0);
        assert!(rows_per_group * approx_row_bytes <= TARGET_ROW_GROUP_BYTES);
    }

    #[test]
    fn wide_rows_get_smaller_row_groups_than_narrow_rows() {
        let wide = random_access_writer_properties(FEATURE_VECTORS_ROW_BYTES).max_row_group_size();
        let narrow = random_access_writer_properties(NESTED_STRUCTS_ROW_BYTES).max_row_group_size();
        assert!(
            wide < narrow,
            "wide {wide} should be smaller than narrow {narrow}"
        );
    }

    #[test]
    fn unsupported_scheme_is_rejected() -> Result<()> {
        assert!(RemoteDataDir::try_new(Url::parse("gs://bucket/prefix/")?).is_err());
        Ok(())
    }
}
