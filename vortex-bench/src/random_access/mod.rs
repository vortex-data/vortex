// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use anyhow::anyhow;
use arrow_array::RecordBatch;
use arrow_array::RecordBatchReader;
use arrow_ipc::writer::FileWriter;
use async_trait::async_trait;
use object_store::ObjectStore;
use object_store::aws::AmazonS3Builder;
use object_store::path::Path as ObjectStorePath;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;
use url::Url;
use vortex::array::ArrayRef;

use crate::Format;
use crate::data_dir;
use crate::idempotent;
use crate::utils::parquet::S3_ROW_GROUP_ROWS;
use crate::utils::parquet::parquet_writer_properties;

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

/// Writer properties for every random-access dataset, the taxi data included: the shared
/// [`parquet_writer_properties`], whose page index and small pages let a lookup read only the
/// pages that hold its rows, in row groups of [`S3_ROW_GROUP_ROWS`]. The row groups are the S3
/// size on local disk too, because the S3 job reads the files that the local job prepares.
pub fn random_access_writer_properties() -> WriterProperties {
    parquet_writer_properties()
        .set_max_row_group_row_count(Some(S3_ROW_GROUP_ROWS))
        .build()
}

/// Write the Parquet file `input` again with [`random_access_writer_properties`].
///
/// A published file can lack the page index, and without it parquet-rs must decode every page of
/// a column chunk to read one row.
pub fn write_random_access_parquet(input: &Path, output: &Path) -> Result<()> {
    let reader = ParquetRecordBatchReaderBuilder::try_new(File::open(input)?)?.build()?;
    let mut writer = ArrowWriter::try_new(
        File::create(output)?,
        reader.schema(),
        Some(random_access_writer_properties()),
    )?;
    for batch in reader {
        writer.write(&batch?)?;
    }
    writer.close()?;
    Ok(())
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
    pub fn key(&self, local_path: &Path) -> Result<ObjectStorePath> {
        let relative = local_path.strip_prefix(data_dir()).map_err(|_| {
            anyhow!(
                "{} is not inside the benchmark data directory",
                local_path.display()
            )
        })?;
        let relative = relative
            .to_str()
            .ok_or_else(|| anyhow!("non-UTF-8 data path: {}", local_path.display()))?;
        // `ObjectStorePath` drops the empty segments left by leading or trailing slashes.
        Ok(ObjectStorePath::from(format!(
            "{}/{relative}",
            self.url.path()
        )))
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
    use arrow_array::Int64Array;
    use parquet::basic::Compression;
    use parquet::basic::ZstdLevel;
    use parquet::file::metadata::PageIndexPolicy;
    use parquet::file::metadata::ParquetMetaDataReader;
    use parquet::file::properties::EnabledStatistics;

    use super::*;

    #[test]
    fn rewritten_parquet_has_a_page_index_and_small_pages() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let input = dir.path().join("published.parquet");
        let output = dir.path().join("random-access.parquet");

        // A file without a page index, like the published taxi data.
        let batch = RecordBatch::try_from_iter([(
            "x",
            Arc::new(Int64Array::from_iter_values(0..100_000)) as _,
        )])?;
        let props = WriterProperties::builder()
            .set_statistics_enabled(EnabledStatistics::None)
            .set_offset_index_disabled(true)
            .build();
        let mut writer = ArrowWriter::try_new(File::create(&input)?, batch.schema(), Some(props))?;
        writer.write(&batch)?;
        writer.close()?;
        let published = ParquetMetaDataReader::new()
            .with_page_index_policy(PageIndexPolicy::Optional)
            .parse_and_finish(&File::open(&input)?)?;
        assert!(
            !published
                .page_index()
                .is_some_and(|index| index.has_offset_indexes())
        );

        write_random_access_parquet(&input, &output)?;

        let metadata = ParquetMetaDataReader::new()
            .with_page_index_policy(PageIndexPolicy::Required)
            .parse_and_finish(&File::open(&output)?)?;
        let page_index = metadata
            .page_index()
            .ok_or_else(|| anyhow!("no page index"))?;
        assert!(page_index.column_index(0, 0).is_some());
        let pages = page_index
            .offset_index(0, 0)
            .ok_or_else(|| anyhow!("no offset index"))?
            .page_locations();
        assert!(pages.len() >= 5);
        for pair in pages.windows(2) {
            // parquet-rs checks the 20,000-row limit after every 1,024 rows.
            assert!(pair[1].first_row_index - pair[0].first_row_index <= 20_480);
        }
        Ok(())
    }

    fn remote(url: &str) -> Result<RemoteDataDir> {
        // `from_env` needs no credentials to construct the client.
        RemoteDataDir::try_new(Url::parse(url)?)
    }

    #[test]
    fn key_mirrors_the_local_data_dir_layout() -> Result<()> {
        let local = data_dir().join("random_access/taxi/taxi.vortex");

        assert_eq!(
            remote("s3://bucket/prefix/")?.key(&local)?.as_ref(),
            "prefix/random_access/taxi/taxi.vortex"
        );
        assert_eq!(
            remote("s3://bucket/")?.key(&local)?.as_ref(),
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
        let err = remote("s3://bucket/prefix/")?
            .key(Path::new("/tmp/taxi.vortex"))
            .unwrap_err();
        assert_eq!(
            err.to_string(),
            "/tmp/taxi.vortex is not inside the benchmark data directory"
        );
        Ok(())
    }

    #[test]
    fn unsupported_scheme_is_rejected() -> Result<()> {
        let err = RemoteDataDir::try_new(Url::parse("gs://bucket/prefix/")?).unwrap_err();
        assert_eq!(err.to_string(), "unsupported remote data dir scheme: gs");
        Ok(())
    }

    #[test]
    fn generated_parquet_is_zstd_level_3() -> Result<()> {
        let props = random_access_writer_properties();
        assert_eq!(
            props.compression(&"embedding".into()),
            Compression::ZSTD(ZstdLevel::try_new(3)?)
        );
        Ok(())
    }
}
