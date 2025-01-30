use std::fs::File;
use std::ops::Range;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

use arrow_array::types::Int64Type;
use arrow_array::{
    ArrayRef as ArrowArrayRef, PrimitiveArray as ArrowPrimitiveArray, RecordBatch,
    RecordBatchReader,
};
use arrow_select::concat::concat_batches;
use arrow_select::take::take_record_batch;
use futures::stream;
use itertools::Itertools;
use log::info;
use object_store::ObjectStore;
use parquet::arrow::arrow_reader::{ArrowReaderOptions, ParquetRecordBatchReaderBuilder};
use parquet::arrow::async_reader::{AsyncFileReader, ParquetObjectReader};
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::file::metadata::RowGroupMetaData;
use serde::{Deserialize, Serialize};
use stream::StreamExt;
use vortex::aliases::hash_map::HashMap;
use vortex::array::ChunkedArray;
use vortex::arrow::FromArrowType;
use vortex::buffer::Buffer;
use vortex::compress::CompressionStrategy;
use vortex::dtype::DType;
use vortex::error::VortexResult;
use vortex::file::{Scan, VortexOpenOptions, VortexWriteOptions};
use vortex::io::{ObjectStoreReadAt, TokioFile, VortexReadAt, VortexWrite};
use vortex::sampling_compressor::{SamplingCompressor, ALL_ENCODINGS_CONTEXT};
use vortex::stream::ArrayStreamExt;
use vortex::{Array, IntoArray, IntoCanonical};

pub const BATCH_SIZE: usize = 65_536;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VortexFooter {
    pub byte_offsets: Vec<u64>,
    pub row_offsets: Vec<u64>,
    pub dtype_range: Range<u64>,
}

pub async fn open_vortex(path: &Path) -> VortexResult<Array> {
    let file = TokioFile::open(path).unwrap();

    VortexOpenOptions::new(ALL_ENCODINGS_CONTEXT.clone())
        .open(file)
        .await?
        .scan(Scan::all())?
        .into_array_data()
        .await
}

pub async fn rewrite_parquet_as_vortex<W: VortexWrite>(
    parquet_path: PathBuf,
    write: W,
) -> VortexResult<()> {
    let chunked = compress_parquet_to_vortex(parquet_path.as_path())?;

    VortexWriteOptions::default()
        .write(write, chunked.into_array_stream())
        .await?;

    Ok(())
}

pub fn read_parquet_to_vortex<P: AsRef<Path>>(parquet_path: P) -> VortexResult<ChunkedArray> {
    let pq_file = File::open(parquet_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(pq_file)?;
    // FIXME(ngates): #157 the compressor should handle batch size.
    let reader = builder.with_batch_size(BATCH_SIZE).build()?;
    let dtype = DType::from_arrow(reader.schema());
    let chunks = reader
        .map(|batch_result| batch_result.unwrap())
        .map(Array::try_from)
        .collect::<VortexResult<Vec<_>>>()?;
    ChunkedArray::try_new(chunks, dtype)
}

pub fn compress_parquet_to_vortex(parquet_path: &Path) -> VortexResult<Array> {
    let chunked = read_parquet_to_vortex(parquet_path)?;
    CompressionStrategy::compress(&SamplingCompressor::default(), &chunked.into_array())
}

pub fn write_csv_as_parquet(csv_path: PathBuf, output_path: &Path) -> VortexResult<()> {
    info!(
        "Compressing {} to parquet",
        csv_path.as_path().to_str().unwrap()
    );
    Command::new("duckdb")
        .arg("-c")
        .arg(format!(
            "COPY (SELECT * FROM read_csv('{}', delim = '|', header = false, nullstr = 'null')) TO '{}' (COMPRESSION ZSTD);",
            csv_path.as_path().to_str().unwrap(),
            output_path.to_str().unwrap()
        ))
        .status()
        .unwrap()
        .exit_ok()
        .unwrap();
    Ok(())
}

async fn take_vortex<T: VortexReadAt + Unpin + 'static>(
    reader: T,
    indices: Buffer<u64>,
) -> VortexResult<Array> {
    VortexOpenOptions::new(ALL_ENCODINGS_CONTEXT.clone())
        .open(reader)
        .await?
        .scan(Scan::all().with_row_indices(indices))?
        .into_array_data()
        .await?
        // For equivalence.... we decompress to make sure we're not cheating too much.
        .into_canonical()
        .map(Array::from)
}

pub async fn take_vortex_object_store(
    fs: Arc<dyn ObjectStore>,
    path: object_store::path::Path,
    indices: Buffer<u64>,
) -> VortexResult<Array> {
    take_vortex(ObjectStoreReadAt::new(fs.clone(), path), indices).await
}

pub async fn take_vortex_tokio(path: &Path, indices: Buffer<u64>) -> VortexResult<Array> {
    take_vortex(TokioFile::open(path)?, indices).await
}

pub async fn take_parquet_object_store(
    fs: Arc<dyn ObjectStore>,
    path: &object_store::path::Path,
    indices: Buffer<u64>,
) -> VortexResult<RecordBatch> {
    let meta = fs.head(path).await?;
    let reader = ParquetObjectReader::new(fs, meta);
    parquet_take_from_stream(reader, indices).await
}

pub async fn take_parquet(path: &Path, indices: Buffer<u64>) -> VortexResult<RecordBatch> {
    let file = tokio::fs::File::open(path).await?;
    parquet_take_from_stream(file, indices).await
}

async fn parquet_take_from_stream<T: AsyncFileReader + Unpin + Send + 'static>(
    async_reader: T,
    indices: Buffer<u64>,
) -> VortexResult<RecordBatch> {
    let builder = ParquetRecordBatchStreamBuilder::new_with_options(
        async_reader,
        ArrowReaderOptions::new().with_page_index(true),
    )
    .await?;

    // We figure out which row groups we need to read and a selection filter for each of them.
    let mut row_groups = HashMap::new();
    let mut row_group_offsets = vec![0];
    row_group_offsets.extend(
        builder
            .metadata()
            .row_groups()
            .iter()
            .map(RowGroupMetaData::num_rows)
            .scan(0i64, |acc, x| {
                *acc += x;
                Some(*acc)
            }),
    );

    for idx in indices {
        let row_group_idx = row_group_offsets
            .binary_search(&(idx as i64))
            .unwrap_or_else(|e| e - 1);
        row_groups
            .entry(row_group_idx)
            .or_insert_with(Vec::new)
            .push((idx as i64) - row_group_offsets[row_group_idx]);
    }
    let row_group_indices = row_groups
        .keys()
        .sorted()
        .map(|i| row_groups[i].clone())
        .collect_vec();

    let reader = builder
        .with_row_groups(row_groups.keys().copied().collect_vec())
        // FIXME(ngates): our indices code assumes the batch size == the row group sizes
        .with_batch_size(10_000_000)
        .build()
        .unwrap();

    let schema = reader.schema().clone();

    let batches = reader
        .enumerate()
        .map(|(idx, batch)| {
            let batch = batch.unwrap();
            let indices = ArrowPrimitiveArray::<Int64Type>::from(row_group_indices[idx].clone());
            let indices_array: ArrowArrayRef = Arc::new(indices);
            take_record_batch(&batch, &indices_array).unwrap()
        })
        .collect::<Vec<_>>()
        .await;

    Ok(concat_batches(&schema, &batches)?)
}
