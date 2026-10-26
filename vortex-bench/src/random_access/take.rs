// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::fs::File as StdFile;
use std::ops::Range;
use std::path::Path;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Context;
use arrow_array::PrimitiveArray;
use arrow_array::RecordBatch;
use arrow_array::cast::AsArray;
use arrow_array::types::Int64Type;
use arrow_ipc::reader::FileReader;
use arrow_select::concat::concat_batches;
use arrow_select::take::take_record_batch;
use async_trait::async_trait;
use futures::TryStreamExt;
use itertools::Itertools;
use object_store::ObjectStoreExt;
use object_store::local::LocalFileSystem;
use object_store::path::Path as ObjectStorePath;
use parking_lot::Mutex;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::arrow_reader::RowSelection;
use parquet::arrow::async_reader;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::file::metadata::PageIndexPolicy;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::stream::ArrayStreamAdapter;
use vortex::array::stream::ArrayStreamExt;
use vortex::arrow::ArrowSessionExt;
use vortex::buffer::Buffer;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::layout::scan::v2::ScanBuilder;
use vortex::scan::strict_sorted_buffer::StrictSortedBuffer;

use crate::Format;
use crate::SESSION;
use crate::random_access::ARROW_ROW_OFFSETS_METADATA_KEY;
use crate::random_access::RandomAccessor;
use crate::random_access::RandomAccessorRet;
use crate::random_access::RemoteDataDir;

/// Random accessor for uncompressed Arrow IPC files.
pub struct ArrowIpcRandomAccessor {
    name: String,
    reader: Mutex<FileReader<StdFile>>,
    row_offsets: Vec<u64>,
    schema: arrow_schema::SchemaRef,
}

impl ArrowIpcRandomAccessor {
    pub fn open(path: PathBuf, name: impl Into<String>) -> anyhow::Result<Self> {
        let reader = FileReader::try_new(StdFile::open(path)?, None)?;
        let row_offsets = reader
            .custom_metadata()
            .get(ARROW_ROW_OFFSETS_METADATA_KEY)
            .context("Arrow IPC file is missing row-offset metadata")?
            .split(',')
            .map(str::parse)
            .collect::<Result<Vec<u64>, _>>()?;
        anyhow::ensure!(
            row_offsets.len() == reader.num_batches() + 1,
            "Arrow IPC row-offset metadata does not match the record batches"
        );
        anyhow::ensure!(
            row_offsets.first() == Some(&0),
            "Arrow IPC row offsets must start at zero"
        );
        anyhow::ensure!(
            row_offsets.windows(2).all(|window| window[0] <= window[1]),
            "Arrow IPC row offsets must be sorted"
        );
        let schema = reader.schema();
        Ok(Self {
            name: name.into(),
            reader: Mutex::new(reader),
            row_offsets,
            schema,
        })
    }
}

#[async_trait]
impl RandomAccessor for ArrowIpcRandomAccessor {
    fn format(&self) -> Format {
        Format::ArrowIpc
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn take(&self, indices: &[u64]) -> anyhow::Result<RandomAccessorRet> {
        let mut by_batch = BTreeMap::<usize, Vec<i64>>::new();
        for &index in indices {
            let batch_idx = self.row_offsets.partition_point(|offset| *offset <= index);
            anyhow::ensure!(
                batch_idx > 0 && batch_idx < self.row_offsets.len(),
                "Arrow row index {index} is out of bounds"
            );
            let batch_idx = batch_idx - 1;
            by_batch
                .entry(batch_idx)
                .or_default()
                .push(i64::try_from(index - self.row_offsets[batch_idx])?);
        }

        let mut reader = self.reader.lock();
        let mut batches = Vec::with_capacity(by_batch.len());
        for (batch_idx, local_indices) in by_batch {
            reader.set_index(batch_idx)?;
            let batch = reader
                .next()
                .context("Arrow IPC record batch is missing")??;
            let indices = PrimitiveArray::<Int64Type>::from(local_indices);
            batches.push(take_record_batch(&batch, &indices)?);
        }

        Ok(RandomAccessorRet::RecordBatch(concat_batches(
            &self.schema,
            &batches,
        )?))
    }
}

/// Random accessor for Vortex format files.
///
/// The file handle is opened at construction time and reused across `take()` calls, sharing its
/// V2 [`ScanCache`](vortex::layout::scan::v2::ScanCache). Rows are read with the V2 scan, which
/// requires files without list layouts.
/// Set `VORTEX_BENCH_SCAN=v1` to compare with V1 using the same open options.
pub struct VortexRandomAccessor {
    name: String,
    format: Format,
    file: VortexFile,
    v2: bool,
}

impl VortexRandomAccessor {
    /// Open a Vortex file and return a ready-to-use accessor.
    pub async fn open(
        path: impl AsRef<Path>,
        name: impl Into<String>,
        format: Format,
    ) -> anyhow::Result<Self> {
        let file = SESSION
            .open_options()
            .with_layout_reader_cache()
            .open_path(path.as_ref())
            .await?;
        Ok(Self {
            name: name.into(),
            format,
            v2: !std::env::var("VORTEX_BENCH_SCAN").is_ok_and(|scan| scan == "v1"),
            file,
        })
    }

    /// Open a Vortex file stored in an object store and return a ready-to-use accessor.
    pub async fn open_object_store(
        remote: &RemoteDataDir,
        path: &Path,
        name: impl Into<String>,
        format: Format,
    ) -> anyhow::Result<Self> {
        let mut options = SESSION.open_options().with_layout_reader_cache();
        let key = remote.key(path)?;
        // Reproduce the former HEAD + bounded footer GET for paired transport measurements.
        if std::env::var("VORTEX_BENCH_SIZE_THEN_READ").is_ok_and(|value| value == "1") {
            options = options.with_file_size(remote.store().head(&key).await?.size);
        }
        let file = options.open_object_store(remote.store(), key).await?;
        Ok(Self {
            name: name.into(),
            format,
            v2: !std::env::var("VORTEX_BENCH_SCAN").is_ok_and(|scan| scan == "v1"),
            file,
        })
    }
}

#[async_trait]
impl RandomAccessor for VortexRandomAccessor {
    fn format(&self) -> Format {
        self.format
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn take(&self, indices: &[u64]) -> anyhow::Result<RandomAccessorRet> {
        let indices = StrictSortedBuffer::try_new(Buffer::from(indices.to_vec()))?;
        let array = if self.v2 {
            let stream = ScanBuilder::new(SESSION.clone(), self.file.scan_file())
                .with_row_indices(indices.clone())
                .with_subsegment_reads(
                    !std::env::var("VORTEX_BENCH_SUBSEGMENTS").is_ok_and(|value| value == "0"),
                )
                .into_stream()?;
            ArrayStreamAdapter::new(self.file.dtype().clone(), stream)
                .read_all()
                .await?
        } else {
            self.file
                .scan()?
                .with_row_indices(indices.clone())
                .into_array_stream()?
                .read_all()
                .await?
        };

        let reference = if std::env::var("VORTEX_BENCH_VERIFY").is_ok_and(|value| value == "1") {
            let stream = ScanBuilder::new(SESSION.clone(), self.file.scan_file())
                .with_row_indices(indices)
                .with_subsegment_reads(false)
                .into_stream()?;
            Some(
                ArrayStreamAdapter::new(self.file.dtype().clone(), stream)
                    .read_all()
                    .await?,
            )
        } else {
            None
        };
        let mut ctx = SESSION.create_execution_ctx();
        if reference.is_some()
            || std::env::var("VORTEX_BENCH_MATERIALIZE").is_ok_and(|value| value == "arrow")
        {
            let arrow = SESSION.arrow().execute_arrow(array, None, &mut ctx)?;
            let batch = RecordBatch::from(arrow.as_struct().clone());
            if let Some(reference) = reference {
                let expected = SESSION.arrow().execute_arrow(reference, None, &mut ctx)?;
                anyhow::ensure!(
                    batch == RecordBatch::from(expected.as_struct().clone()),
                    "Sparse read differs from whole-segment reference"
                );
            }
            return Ok(RandomAccessorRet::RecordBatch(batch));
        }
        // Keep the historical default: canonicalize the outer array, leaving nested values lazy.
        let canonical = array.execute::<Canonical>(&mut ctx)?.into_array();
        Ok(RandomAccessorRet::ArrayRef(canonical))
    }
}

/// Random accessor for Parquet format files.
///
/// Parquet footer, page index and row group offsets are parsed at construction time and reused to
/// map indices to row groups in each `take()` call.
///
/// Each take reads only the rows it asks for: the indices of every touched row group become a
/// [`RowSelection`], which together with the offset index lets parquet-rs fetch and decode only
/// the pages holding selected rows. Row groups are read concurrently, one task each, and their
/// rows concatenated in order.
///
/// Local files are read through `object_store`'s [`LocalFileSystem`], as DataFusion reads them,
/// so the pages of one fetch are read in a single blocking call rather than one call each.
pub struct ParquetRandomAccessor {
    name: String,
    /// Cumulative row offsets per row group (length = num_row_groups + 1).
    row_group_offsets: Vec<u64>,
    /// Cached Arrow reader metadata (footer) to avoid re-parsing on each take.
    arrow_metadata: ArrowReaderMetadata,
    /// Reader for the file, cloned for every row group read.
    reader: ObjectReader,
}

/// Reader for a Parquet file held in an object store.
#[expect(
    deprecated,
    reason = "arrow-rs deprecated this in favour of a hand-rolled AsyncFileReader; \
        keeping it holds the measured I/O path fixed (arrow-rs#10308)"
)]
type ObjectReader = async_reader::ParquetObjectReader;

impl ParquetRandomAccessor {
    /// Open a Parquet file, parse the footer, and return a ready-to-use accessor.
    pub async fn open(path: PathBuf, name: impl Into<String>) -> anyhow::Result<Self> {
        let location = ObjectStorePath::from_absolute_path(path.canonicalize()?)?;
        Self::open_reader(
            ObjectReader::new(Arc::new(LocalFileSystem::new()), location),
            name,
        )
        .await
    }

    /// Open a Parquet file stored in an object store and return a ready-to-use accessor.
    pub async fn open_object_store(
        remote: &RemoteDataDir,
        path: &Path,
        name: impl Into<String>,
    ) -> anyhow::Result<Self> {
        Self::open_reader(
            ObjectReader::new(Arc::clone(remote.store()), remote.key(path)?),
            name,
        )
        .await
    }

    async fn open_reader(
        mut reader: ObjectReader,
        name: impl Into<String>,
    ) -> anyhow::Result<Self> {
        let arrow_metadata = load_metadata(&mut reader).await?;
        Self::new(name, arrow_metadata, reader)
    }

    fn new(
        name: impl Into<String>,
        arrow_metadata: ArrowReaderMetadata,
        reader: ObjectReader,
    ) -> anyhow::Result<Self> {
        let mut row_group_offsets = vec![0];
        let mut total = 0u64;
        for row_group in arrow_metadata.metadata().row_groups() {
            total += u64::try_from(row_group.num_rows())?;
            row_group_offsets.push(total);
        }

        Ok(Self {
            name: name.into(),
            row_group_offsets,
            arrow_metadata,
            reader,
        })
    }
}

/// Parse the Parquet footer, including the page index, from any async reader.
async fn load_metadata<T: AsyncFileReader + Send>(
    reader: &mut T,
) -> anyhow::Result<ArrowReaderMetadata> {
    let options = ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Required);
    Ok(ArrowReaderMetadata::load_async(reader, options).await?)
}

/// The rows to read from one row group: runs of consecutive row-group-local indices.
struct RowGroupRead {
    row_group: usize,
    num_rows: usize,
    ranges: Vec<Range<usize>>,
}

impl RowGroupRead {
    fn selected_rows(&self) -> usize {
        self.ranges.iter().map(ExactSizeIterator::len).sum()
    }
}

/// Group strictly sorted file-level `indices` into per-row-group runs of consecutive rows.
fn plan_row_group_reads(
    row_group_offsets: &[u64],
    indices: &[u64],
) -> anyhow::Result<Vec<RowGroupRead>> {
    anyhow::ensure!(
        indices.is_sorted_by(|a, b| a < b),
        "Parquet take indices must be strictly sorted"
    );
    let mut reads: Vec<RowGroupRead> = Vec::new();
    for &index in indices {
        let row_group = row_group_offsets.partition_point(|offset| *offset <= index);
        anyhow::ensure!(
            row_group > 0 && row_group < row_group_offsets.len(),
            "Parquet row index {index} is out of bounds"
        );
        let row_group = row_group - 1;
        let local = usize::try_from(index - row_group_offsets[row_group])?;

        if reads.last().is_none_or(|read| read.row_group != row_group) {
            reads.push(RowGroupRead {
                row_group,
                num_rows: usize::try_from(
                    row_group_offsets[row_group + 1] - row_group_offsets[row_group],
                )?,
                ranges: Vec::new(),
            });
        }
        let last = reads.len() - 1;
        let ranges = &mut reads[last].ranges;
        match ranges.last_mut() {
            Some(range) if range.end == local => range.end += 1,
            _ => ranges.push(local..local + 1),
        }
    }
    Ok(reads)
}

#[async_trait]
impl RandomAccessor for ParquetRandomAccessor {
    fn format(&self) -> Format {
        Format::Parquet
    }

    fn name(&self) -> &str {
        &self.name
    }

    async fn take(&self, indices: &[u64]) -> anyhow::Result<RandomAccessorRet> {
        let reads = plan_row_group_reads(&self.row_group_offsets, indices)?
            .into_iter()
            .map(|read| {
                let reader = self.reader.clone();
                let metadata = self.arrow_metadata.clone();
                tokio::spawn(async move {
                    let batch_size = read.selected_rows();
                    let selection = RowSelection::from_consecutive_ranges(
                        read.ranges.into_iter(),
                        read.num_rows,
                    );
                    ParquetRecordBatchStreamBuilder::new_with_metadata(reader, metadata)
                        .with_row_groups(vec![read.row_group])
                        .with_row_selection(selection)
                        .with_batch_size(batch_size)
                        .build()?
                        .try_collect::<Vec<_>>()
                        .await
                        .map_err(anyhow::Error::from)
                })
            })
            .collect_vec();

        let mut batches = Vec::new();
        for read in reads {
            batches.extend(read.await??);
        }

        let schema = Arc::clone(self.arrow_metadata.schema());
        Ok(RandomAccessorRet::RecordBatch(concat_batches(
            &schema, &batches,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use arrow_array::Int64Array;
    use arrow_array::RecordBatch;
    use arrow_ipc::writer::FileWriter;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use arrow_schema::Schema;
    use parquet::arrow::ArrowWriter;
    use parquet::file::properties::WriterProperties;

    use super::*;

    #[tokio::test]
    async fn arrow_ipc_random_accessor_takes_rows_across_record_batches() -> anyhow::Result<()> {
        let file = tempfile::NamedTempFile::new()?;
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        {
            let mut writer = FileWriter::try_new(file.reopen()?, schema.as_ref())?;
            writer.write(&RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from(vec![0, 1]))],
            )?)?;
            writer.write(&RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from(vec![2, 3, 4]))],
            )?)?;
            writer.write_metadata(ARROW_ROW_OFFSETS_METADATA_KEY, "0,2,5");
            writer.finish()?;
        }

        let accessor = ArrowIpcRandomAccessor::open(file.path().to_path_buf(), "arrow-ipc")?;
        let RandomAccessorRet::RecordBatch(actual) = accessor.take(&[1, 3, 4]).await? else {
            anyhow::bail!("Arrow accessor returned a Vortex array")
        };
        let expected =
            RecordBatch::try_new(schema, vec![Arc::new(Int64Array::from(vec![1, 3, 4]))])?;
        assert_eq!(actual, expected);
        Ok(())
    }

    #[test]
    #[expect(
        clippy::single_range_in_vec_init,
        reason = "row group 2 reads one range"
    )]
    fn parquet_reads_are_planned_as_consecutive_runs_per_row_group() -> anyhow::Result<()> {
        let reads = plan_row_group_reads(&[0, 10, 20, 25], &[1, 2, 3, 7, 10, 19, 24])?;
        let plan = reads
            .iter()
            .map(|read| (read.row_group, read.num_rows, read.ranges.clone()))
            .collect_vec();
        assert_eq!(
            plan,
            vec![
                (0, 10, vec![1..4, 7..8]),
                (1, 10, vec![0..1, 9..10]),
                (2, 5, vec![4..5]),
            ]
        );
        assert!(plan_row_group_reads(&[0, 10], &[10]).is_err());
        assert!(plan_row_group_reads(&[0, 10], &[3, 3]).is_err());
        Ok(())
    }

    #[tokio::test]
    async fn parquet_random_accessor_takes_rows_across_row_groups_and_pages() -> anyhow::Result<()>
    {
        let file = tempfile::NamedTempFile::new()?;
        let schema = Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        {
            let props = WriterProperties::builder()
                .set_max_row_group_row_count(Some(100))
                .set_data_page_row_count_limit(10)
                .set_write_batch_size(10)
                .build();
            let mut writer =
                ArrowWriter::try_new(file.reopen()?, Arc::clone(&schema), Some(props))?;
            writer.write(&RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from_iter_values(0..250))],
            )?)?;
            writer.close()?;
        }

        let accessor = ParquetRandomAccessor::open(file.path().to_path_buf(), "parquet").await?;
        let indices = [0, 1, 2, 55, 99, 100, 101, 180, 249];
        let RandomAccessorRet::RecordBatch(actual) = accessor.take(&indices).await? else {
            anyhow::bail!("Parquet accessor returned a Vortex array")
        };
        let expected = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int64Array::from(vec![
                0, 1, 2, 55, 99, 100, 101, 180, 249,
            ]))],
        )?;
        assert_eq!(actual, expected);
        Ok(())
    }
}
