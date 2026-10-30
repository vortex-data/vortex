// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::Write;
use std::path::Path;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use anyhow::bail;
use arrow_array::RecordBatch;
use arrow_schema::DataType;
use arrow_schema::Schema;
use async_trait::async_trait;
use bytes::Bytes;
use parking_lot::Mutex;
use parquet::arrow::ArrowWriter;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ArrowReaderMetadata;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowColumnChunk;
use parquet::arrow::arrow_writer::ArrowColumnWriter;
use parquet::arrow::arrow_writer::ArrowLeafColumn;
use parquet::arrow::arrow_writer::ArrowRowGroupWriterFactory;
use parquet::arrow::arrow_writer::compute_leaves;
use parquet::basic::Compression;
use parquet::basic::ZstdLevel;
use parquet::file::metadata::RowGroupMetaData;
use parquet::file::properties::DEFAULT_MAX_ROW_GROUP_ROW_COUNT;
use parquet::file::properties::WriterProperties;
use tokio::sync::mpsc;
use tokio::task::JoinHandle;
use vortex::error::VortexExpect;
use vortex_bench::Format;
use vortex_bench::compress::Compressed;
use vortex_bench::compress::CompressedData;
use vortex_bench::compress::Compressor;
use vortex_bench::compress::Uncompressed;
use vortex_bench::compress::read_projection;

/// Compressor implementation for Parquet format with ZSTD compression.
pub struct ParquetCompressor {
    compression: Compression,
}

impl ParquetCompressor {
    pub fn new() -> Self {
        Self {
            compression: Compression::ZSTD(
                ZstdLevel::try_new(3)
                    .ok()
                    .vortex_expect("3 is a valid Zstd compression level"),
            ),
        }
    }

    pub fn with_compression(compression: Compression) -> Self {
        Self { compression }
    }
}

impl Default for ParquetCompressor {
    fn default() -> Self {
        Self::new()
    }
}

/// Return the Arrow memory size after decoding the input Parquet file.
pub fn arrow_uncompressed_size(parquet_path: &Path) -> anyhow::Result<u64> {
    let file = File::open(parquet_path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut total = 0u64;

    for batch in reader {
        let batch = batch?;
        let batch_size = u64::try_from(batch.get_array_memory_size())?;
        total = total
            .checked_add(batch_size)
            .ok_or_else(|| anyhow::anyhow!("Arrow memory size exceeds u64"))?;
    }

    Ok(total)
}

#[async_trait]
impl Compressor for ParquetCompressor {
    fn format(&self) -> Format {
        Format::Parquet
    }

    async fn load(&self, parquet_path: &Path) -> anyhow::Result<Uncompressed> {
        Uncompressed::read_arrow(parquet_path)
    }

    async fn compress(&self, input: &Uncompressed) -> anyhow::Result<Compressed> {
        let (schema, batches) = input.arrow()?;
        let (schema, batches) = (Arc::clone(schema), batches.to_vec());
        let compression = self.compression;

        // Run the timed work as a task so it uses only the runtime's worker threads.
        let start = Instant::now();
        let (buf, size) = tokio::spawn(async move {
            let mut buf = Vec::new();
            let size = parquet_compress_write(batches, schema, compression, &mut buf).await?;
            anyhow::Ok((buf, size))
        })
        .await??;
        let elapsed = start.elapsed();

        Ok(Compressed {
            data: CompressedData::Bytes(Bytes::from(buf)),
            size: size as u64,
            elapsed,
        })
    }

    async fn decompress(&self, compressed: &Compressed) -> anyhow::Result<Duration> {
        let CompressedData::Bytes(buf) = &compressed.data else {
            bail!("Parquet decompression expects in-memory bytes");
        };
        let buf = buf.clone();

        let timer = Instant::now();
        tokio::spawn(parquet_decompress_read(buf)).await??;
        Ok(timer.elapsed())
    }
}

/// Write `batches` to `buf` as one Parquet file the way DataFusion's `ParquetSink` does with its
/// default options, returning the bytes written before the footer.
///
/// Batches are split at the writer's `max_row_group_row_count`, the same row-group boundaries a
/// sequential `ArrowWriter` produces. Every leaf column of the open row group is encoded by its own
/// task, fed through a channel holding at most `MAX_BUFFERED_BATCHES` arrays. When a row group is
/// full, a separate task joins its column writers while the next row group starts, and at most
/// `MAX_PARALLEL_ROW_GROUPS` finished row groups wait to be appended to the file in order.
pub async fn parquet_compress_write(
    batches: Vec<RecordBatch>,
    schema: Arc<Schema>,
    compression: Compression,
    buf: &mut Vec<u8>,
) -> anyhow::Result<usize> {
    let writer_properties = WriterProperties::builder()
        .set_compression(compression)
        .build();
    let max_rows = writer_properties
        .max_row_group_row_count()
        .unwrap_or(DEFAULT_MAX_ROW_GROUP_ROW_COUNT);
    let output = SharedBuffer::default();
    // `ArrowWriter` embeds the Arrow schema in the footer; converting it into the lower-level
    // writer keeps that while letting each column chunk be encoded as its own task.
    let (mut writer, factory) =
        ArrowWriter::try_new(output.clone(), Arc::clone(&schema), Some(writer_properties))?
            .into_serialized_writer()?;

    let (row_group_tx, mut row_group_rx) = mpsc::channel(MAX_PARALLEL_ROW_GROUPS);
    let serialize = tokio::spawn(serialize_row_groups(
        batches,
        schema,
        factory,
        max_rows,
        row_group_tx,
    ));

    // Move the encoded bytes out of the shared buffer as they pass `BUFFER_FLUSH_BYTES`, as
    // DataFusion streams them to the object store, so no buffer grows to the size of the file.
    let mut parts = Vec::new();
    while let Some(row_group) = row_group_rx.recv().await {
        let chunks = row_group.await??;
        let mut row_group_writer = writer.next_row_group()?;
        for chunk in chunks {
            chunk.append_to_row_group(&mut row_group_writer)?;
            output.flush_into(&mut parts, BUFFER_FLUSH_BYTES);
        }
        row_group_writer.close()?;
    }
    serialize.await??;

    let n_bytes = writer.bytes_written();
    writer.close()?;
    output.flush_into(&mut parts, 0);

    buf.reserve_exact(parts.iter().map(Vec::len).sum());
    for part in parts {
        buf.extend_from_slice(&part);
    }
    Ok(n_bytes)
}

type ColumnTask = JoinHandle<anyhow::Result<ArrowColumnWriter>>;
type RowGroupTask = JoinHandle<anyhow::Result<Vec<ArrowColumnChunk>>>;

/// Feed `batches` to per-column encoding tasks, sending a task that finishes each full row group
/// to `row_group_tx`.
async fn serialize_row_groups(
    batches: Vec<RecordBatch>,
    schema: Arc<Schema>,
    factory: ArrowRowGroupWriterFactory,
    max_rows: usize,
    row_group_tx: mpsc::Sender<RowGroupTask>,
) -> anyhow::Result<()> {
    let mut row_group_idx = 0;
    let (mut column_tasks, mut column_txs) =
        spawn_column_writers(factory.create_column_writers(row_group_idx)?);
    let mut row_group_rows = 0;

    for mut batch in batches {
        loop {
            let len = (max_rows - row_group_rows).min(batch.num_rows());
            send_leaves(&column_txs, &batch.slice(0, len), &schema).await?;
            row_group_rows += len;
            if row_group_rows < max_rows {
                break;
            }

            // Closing the channels lets the column tasks finish. Join them on their own task so
            // the next row group starts encoding without waiting for this one.
            drop(column_txs);
            let finish = tokio::spawn(finish_row_group(column_tasks));
            if row_group_tx.send(finish).await.is_err() {
                // The receiver failed and will report its own error.
                return Ok(());
            }
            row_group_idx += 1;
            row_group_rows = 0;
            (column_tasks, column_txs) =
                spawn_column_writers(factory.create_column_writers(row_group_idx)?);

            batch = batch.slice(len, batch.num_rows() - len);
            if batch.num_rows() == 0 {
                break;
            }
        }
    }

    drop(column_txs);
    if row_group_rows > 0 {
        let finish = tokio::spawn(finish_row_group(column_tasks));
        // A send error means the receiver failed and will report its own error.
        drop(row_group_tx.send(finish).await);
    }
    Ok(())
}

/// Spawn one task per leaf column writer, each encoding the arrays it receives on its channel.
fn spawn_column_writers(
    writers: Vec<ArrowColumnWriter>,
) -> (Vec<ColumnTask>, Vec<mpsc::Sender<ArrowLeafColumn>>) {
    writers
        .into_iter()
        .map(|mut writer| {
            let (tx, mut rx) = mpsc::channel::<ArrowLeafColumn>(MAX_BUFFERED_BATCHES);
            let task = tokio::spawn(async move {
                while let Some(leaf) = rx.recv().await {
                    writer.write(&leaf)?;
                }
                anyhow::Ok(writer)
            });
            (task, tx)
        })
        .unzip()
}

/// Send each leaf column of `batch` to its column writer task.
async fn send_leaves(
    column_txs: &[mpsc::Sender<ArrowLeafColumn>],
    batch: &RecordBatch,
    schema: &Schema,
) -> anyhow::Result<()> {
    let mut column_txs = column_txs.iter();
    for (array, field) in batch.columns().iter().zip(schema.fields()) {
        for (leaf, tx) in compute_leaves(field, array)?
            .into_iter()
            .zip(column_txs.by_ref())
        {
            // A closed channel means the column task failed; joining it reports the error.
            if tx.send(leaf).await.is_err() {
                return Ok(());
            }
        }
    }
    Ok(())
}

/// Wait for a row group's column tasks and close their writers into encoded column chunks.
async fn finish_row_group(column_tasks: Vec<ColumnTask>) -> anyhow::Result<Vec<ArrowColumnChunk>> {
    let mut chunks = Vec::with_capacity(column_tasks.len());
    for task in column_tasks {
        chunks.push(task.await??.close()?);
    }
    Ok(chunks)
}

/// The writer's output buffer, shared so encoded bytes can be moved out while the file writer
/// still holds it.
#[derive(Clone, Default)]
struct SharedBuffer(Arc<Mutex<Vec<u8>>>);

impl SharedBuffer {
    /// Move the buffered bytes into `parts` if there are more than `min_bytes` of them.
    fn flush_into(&self, parts: &mut Vec<Vec<u8>>, min_bytes: usize) {
        let mut buffer = self.0.lock();
        if buffer.len() > min_bytes {
            let capacity = buffer.capacity();
            parts.push(std::mem::replace(
                &mut *buffer,
                Vec::with_capacity(capacity),
            ));
        }
    }
}

impl Write for SharedBuffer {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().write(buf)
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// Scan every row group of an in-memory Parquet file the way DataFusion does with its default
/// options, returning the in-memory size of the decoded batches.
///
/// The file's bytes are split into one range per runtime worker (DataFusion's
/// `target_partitions`), and each row group belongs to the range holding its first page. Each
/// non-empty range is one task decoding its row groups in order, in batches of
/// `DECODE_BATCH_SIZE` rows, with strings and binaries read as view arrays.
pub async fn parquet_decompress_read(buf: Bytes) -> anyhow::Result<usize> {
    let metadata = ArrowReaderMetadata::load(&buf, ArrowReaderOptions::default())?;
    // Decode strings and binaries to view arrays, as DataFusion's `schema_force_view_types`
    // does; they are also the Arrow types the Vortex reader produces.
    let view_schema = Arc::new(Schema::new_with_metadata(
        metadata
            .schema()
            .fields()
            .iter()
            .map(|field| {
                let data_type = match field.data_type() {
                    DataType::Utf8 | DataType::LargeUtf8 => DataType::Utf8View,
                    DataType::Binary | DataType::LargeBinary => DataType::BinaryView,
                    other => other.clone(),
                };
                field.as_ref().clone().with_data_type(data_type)
            })
            .collect::<Vec<_>>(),
        metadata.schema().metadata().clone(),
    ));
    let metadata = ArrowReaderMetadata::try_new(
        Arc::clone(metadata.metadata()),
        ArrowReaderOptions::new().with_schema(view_schema),
    )?;
    let num_columns = metadata.schema().fields().len();
    let columns =
        read_projection(num_columns).map_or_else(|| (0..num_columns).collect(), <[usize]>::to_vec);
    let projection = ProjectionMask::roots(metadata.parquet_schema(), columns);

    let num_workers = tokio::runtime::Handle::current().metrics().num_workers();
    let partitions = scan_partitions(
        metadata.metadata().row_groups(),
        u64::try_from(buf.len())?,
        num_workers,
    );
    let tasks: Vec<_> = partitions
        .into_iter()
        .map(|row_groups| {
            let reader =
                ParquetRecordBatchReaderBuilder::new_with_metadata(buf.clone(), metadata.clone())
                    .with_row_groups(row_groups)
                    .with_projection(projection.clone())
                    .with_batch_size(DECODE_BATCH_SIZE);
            tokio::spawn(async move {
                let mut nbytes = 0;
                for batch in reader.build()? {
                    nbytes += batch?.get_array_memory_size();
                }
                anyhow::Ok(nbytes)
            })
        })
        .collect();

    let mut nbytes = 0;
    for task in tasks {
        nbytes += task.await??;
    }
    Ok(nbytes)
}

/// Group row groups into the scan partitions DataFusion would create for a file of `file_len`
/// bytes with `target_partitions` partitions, dropping empty ones.
///
/// A file smaller than `REPARTITION_MIN_FILE_BYTES` stays one partition. Otherwise the file's
/// bytes are split into `target_partitions` equal ranges, and each row group is read by the range
/// containing the offset of its first column's first page.
fn scan_partitions(
    row_groups: &[RowGroupMetaData],
    file_len: u64,
    target_partitions: usize,
) -> Vec<Vec<usize>> {
    let target_partitions = if file_len < REPARTITION_MIN_FILE_BYTES {
        1
    } else {
        target_partitions.max(1)
    };
    let range_len = file_len.div_ceil(target_partitions as u64).max(1);
    let mut partitions = vec![Vec::new(); target_partitions];
    for (idx, row_group) in row_groups.iter().enumerate() {
        let column = row_group.column(0);
        let offset = column
            .dictionary_page_offset()
            .unwrap_or_else(|| column.data_page_offset());
        let partition = usize::try_from(u64::try_from(offset).unwrap_or(0) / range_len)
            .unwrap_or(usize::MAX)
            .min(target_partitions - 1);
        partitions[partition].push(idx);
    }
    partitions.retain(|row_groups| !row_groups.is_empty());
    partitions
}

/// Leaf arrays queued per column writer task: DataFusion's
/// `maximum_buffered_record_batches_per_stream` default.
const MAX_BUFFERED_BATCHES: usize = 2;

/// Finished row groups queued for appending to the file: DataFusion's
/// `maximum_parallel_row_group_writers` default.
const MAX_PARALLEL_ROW_GROUPS: usize = 1;

/// Output bytes buffered before being moved out of the writer: DataFusion's `BUFFER_FLUSH_BYTES`.
const BUFFER_FLUSH_BYTES: usize = 1024000;

/// Files smaller than this are scanned as one partition: DataFusion's
/// `repartition_file_min_size` default.
const REPARTITION_MIN_FILE_BYTES: u64 = 1024 * 1024;

/// Rows per decoded batch: DataFusion's default `batch_size`, which is also the Vortex writer's
/// row block size, so both readers emit batches of the same length.
const DECODE_BATCH_SIZE: usize = 8192;

#[cfg(test)]
mod tests {
    use arrow_array::Int64Array;
    use arrow_array::StringArray;
    use arrow_schema::DataType;
    use arrow_schema::Field;
    use parquet::file::metadata::ParquetMetaData;
    use rstest::rstest;

    use super::*;

    fn batch(start: i64, len: usize) -> anyhow::Result<RecordBatch> {
        let ints = Int64Array::from_iter_values(start..start + len as i64);
        let strs = StringArray::from_iter_values(ints.values().iter().map(|v| v.to_string()));
        Ok(RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("i", DataType::Int64, false),
                Field::new("s", DataType::Utf8, false),
            ])),
            vec![Arc::new(ints), Arc::new(strs)],
        )?)
    }

    fn row_group_rows(metadata: &ParquetMetaData) -> Vec<i64> {
        metadata
            .row_groups()
            .iter()
            .map(RowGroupMetaData::num_rows)
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn parallel_write_matches_arrow_writer() -> anyhow::Result<()> {
        let rows = DEFAULT_MAX_ROW_GROUP_ROW_COUNT;
        let batches = vec![
            batch(0, rows / 2)?,
            batch(0, rows)?,
            batch(0, 0)?,
            batch(0, 7)?,
        ];
        let schema = batches[0].schema();
        let compression = Compression::ZSTD(ZstdLevel::default());

        let mut parallel = Vec::new();
        parquet_compress_write(
            batches.clone(),
            Arc::clone(&schema),
            compression,
            &mut parallel,
        )
        .await?;
        let parallel = Bytes::from(parallel);

        let mut sequential = Vec::new();
        let props = WriterProperties::builder()
            .set_compression(compression)
            .build();
        let mut writer = ArrowWriter::try_new(&mut sequential, Arc::clone(&schema), Some(props))?;
        for batch in &batches {
            writer.write(batch)?;
        }
        writer.close()?;

        let parallel_meta = ArrowReaderMetadata::load(&parallel, Default::default())?;
        let sequential_meta =
            ArrowReaderMetadata::load(&Bytes::from(sequential), Default::default())?;
        assert_eq!(
            row_group_rows(parallel_meta.metadata()),
            row_group_rows(sequential_meta.metadata())
        );
        assert_eq!(parallel_meta.schema(), sequential_meta.schema());

        let read = ParquetRecordBatchReaderBuilder::try_new(parallel.clone())?
            .build()?
            .collect::<Result<Vec<_>, _>>()?;
        assert_eq!(
            arrow_select::concat::concat_batches(&schema, &read)?,
            arrow_select::concat::concat_batches(&schema, &batches)?
        );
        assert!(parquet_decompress_read(parallel).await? > 0);
        Ok(())
    }

    #[rstest]
    #[case::small_file(1000, 4, vec![vec![0, 1, 2]])]
    #[case::one_partition_per_row_group(200_000, 4, vec![vec![0], vec![1], vec![2]])]
    #[case::shared_partition(200_000, 2, vec![vec![0, 1], vec![2]])]
    #[tokio::test(flavor = "multi_thread", worker_threads = 2)]
    async fn scan_partitions_follow_byte_ranges(
        #[case] rows: usize,
        #[case] target_partitions: usize,
        #[case] expected: Vec<Vec<usize>>,
    ) -> anyhow::Result<()> {
        // Three row groups of incompressible strings, so each covers about a third of the file.
        let batches = vec![batch(0, rows)?, batch(rows as i64, rows)?, batch(0, rows)?];
        let schema = batches[0].schema();
        let props = WriterProperties::builder()
            .set_max_row_group_row_count(Some(rows))
            .set_compression(Compression::UNCOMPRESSED)
            .build();
        let mut buf = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut buf, schema, Some(props))?;
        for batch in &batches {
            writer.write(batch)?;
        }
        writer.close()?;
        let file_len = u64::try_from(buf.len())?;
        let metadata = ArrowReaderMetadata::load(&Bytes::from(buf), Default::default())?;

        assert_eq!(
            scan_partitions(
                metadata.metadata().row_groups(),
                file_len,
                target_partitions
            ),
            expected
        );
        Ok(())
    }
}
