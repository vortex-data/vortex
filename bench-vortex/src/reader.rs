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
use bytes::{Bytes, BytesMut};
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
use vortex::array::{ChunkedArray, PrimitiveArray};
use vortex::arrow::FromArrowType;
use vortex::buffer::Buffer;
use vortex::compress::CompressionStrategy;
use vortex::dtype::DType;
use vortex::error::{vortex_err, VortexResult};
use vortex::sampling_compressor::SamplingCompressor;
use vortex::serde::chunked_reader::ChunkedArrayReader;
use vortex::serde::io::{ObjectStoreExt, VortexReadAt, VortexWrite};
use vortex::serde::stream_reader::StreamArrayReader;
use vortex::serde::stream_writer::StreamArrayWriter;
use vortex::serde::DTypeReader;
use vortex::stream::ArrayStreamExt;
use vortex::{Array, ArrayDType, IntoArray, IntoCanonical};

use crate::{COMPRESSORS, CTX};

pub const BATCH_SIZE: usize = 65_536;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VortexFooter {
    pub byte_offsets: Vec<u64>,
    pub row_offsets: Vec<u64>,
    pub dtype_range: Range<u64>,
}

pub async fn open_vortex(path: &Path) -> VortexResult<Array> {
    let file = tokio::fs::File::open(path).await.unwrap();
    let reader = StreamArrayReader::try_new(file, CTX.clone())
        .await?
        .load_dtype()
        .await?;
    reader
        .into_array_stream()
        .collect_chunked()
        .await
        .map(IntoArray::into_array)
}

pub async fn rewrite_parquet_as_vortex<W: VortexWrite>(
    parquet_path: PathBuf,
    write: W,
) -> VortexResult<()> {
    let chunked = compress_parquet_to_vortex(parquet_path.as_path())?;

    let written = StreamArrayWriter::new(write)
        .write_array_stream(chunked.array_stream())
        .await?;

    let layout = written.array_layouts()[0].clone();
    let mut w = written.into_inner();
    let mut s = flexbuffers::FlexbufferSerializer::new();
    VortexFooter {
        byte_offsets: layout.chunks.byte_offsets,
        row_offsets: layout.chunks.row_offsets,
        dtype_range: layout.dtype.begin..layout.dtype.end,
    }
    .serialize(&mut s)?;
    let footer_bytes = Buffer::Bytes(Bytes::from(s.take_buffer()));
    let footer_len = footer_bytes.len() as u64;
    w.write_all(footer_bytes).await?;
    w.write_all(footer_len.to_le_bytes()).await?;

    Ok(())
}

pub fn read_parquet_to_vortex<P: AsRef<Path>>(parquet_path: P) -> VortexResult<ChunkedArray> {
    let taxi_pq = File::open(parquet_path)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(taxi_pq)?;
    // FIXME(ngates): #157 the compressor should handle batch size.
    let reader = builder.with_batch_size(BATCH_SIZE).build()?;
    let dtype = DType::from_arrow(reader.schema());
    let chunks = reader
        .map(|batch_result| batch_result.unwrap())
        .map(Array::try_from)
        .collect::<VortexResult<Vec<_>>>()?;
    ChunkedArray::try_new(chunks, dtype)
}

pub fn compress_parquet_to_vortex(parquet_path: &Path) -> VortexResult<ChunkedArray> {
    let chunked = read_parquet_to_vortex(parquet_path)?;
    let compressor: &dyn CompressionStrategy = &SamplingCompressor::new(COMPRESSORS.clone());
    let dtype = chunked.dtype().clone();
    ChunkedArray::try_new(
        chunked
            .chunks()
            .map(|x| compressor.compress(&x))
            .collect::<VortexResult<Vec<_>>>()?,
        dtype,
    )
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

pub async fn read_vortex_footer_format<R: VortexReadAt>(
    reader: R,
    len: u64,
) -> VortexResult<ChunkedArrayReader<R>> {
    let mut buf = BytesMut::with_capacity(8);
    unsafe { buf.set_len(8) }
    buf = reader.read_at_into(len - 8, buf).await?;
    let footer_len = u64::from_le_bytes(buf.as_ref().try_into().unwrap()) as usize;

    buf.reserve(footer_len - buf.len());
    unsafe { buf.set_len(footer_len) }
    buf = reader
        .read_at_into(len - footer_len as u64 - 8, buf)
        .await?;

    let footer: VortexFooter = VortexFooter::deserialize(
        flexbuffers::Reader::get_root(buf.as_ref()).map_err(|e| vortex_err!("{}", e))?,
    )?;

    let header_len = (footer.dtype_range.end - footer.dtype_range.start) as usize;
    buf.reserve(header_len - buf.len());
    unsafe { buf.set_len(header_len) }
    buf = reader.read_at_into(footer.dtype_range.start, buf).await?;
    let dtype = DTypeReader::new(buf).await?.read_dtype().await?;

    ChunkedArrayReader::try_new(
        reader,
        CTX.clone(),
        dtype.into(),
        PrimitiveArray::from(footer.byte_offsets).into_array(),
        PrimitiveArray::from(footer.row_offsets).into_array(),
    )
}

pub async fn take_vortex_object_store(
    fs: &Arc<dyn ObjectStore>,
    path: &object_store::path::Path,
    indices: &[u64],
) -> VortexResult<Array> {
    let head = fs.head(path).await?;
    let indices_array = indices.to_vec().into_array();
    let taken = read_vortex_footer_format(fs.vortex_reader(path), head.size as u64)
        .await?
        .take_rows(&indices_array)
        .await?;
    // For equivalence.... we flatten to make sure we're not cheating too much.
    Ok(taken.into_canonical()?.into())
}

pub async fn take_vortex_tokio(path: &Path, indices: &[u64]) -> VortexResult<Array> {
    let len = File::open(path)?.metadata()?.len();
    let indices_array = indices.to_vec().into_array();
    let taken = read_vortex_footer_format(tokio::fs::File::open(path).await?, len)
        .await?
        .take_rows(&indices_array)
        .await?;
    // For equivalence.... we flatten to make sure we're not cheating too much.
    Ok(taken.into_canonical()?.into())
}

pub async fn take_parquet_object_store(
    fs: Arc<dyn ObjectStore>,
    path: &object_store::path::Path,
    indices: &[u64],
) -> VortexResult<RecordBatch> {
    let meta = fs.head(path).await?;
    let reader = ParquetObjectReader::new(fs, meta);
    parquet_take_from_stream(reader, indices).await
}

pub async fn take_parquet(path: &Path, indices: &[u64]) -> VortexResult<RecordBatch> {
    let file = tokio::fs::File::open(path).await?;
    parquet_take_from_stream(file, indices).await
}

async fn parquet_take_from_stream<T: AsyncFileReader + Unpin + Send + 'static>(
    async_reader: T,
    indices: &[u64],
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
            .binary_search(&(*idx as i64))
            .unwrap_or_else(|e| e - 1);
        row_groups
            .entry(row_group_idx)
            .or_insert_with(Vec::new)
            .push((*idx as i64) - row_group_offsets[row_group_idx]);
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
