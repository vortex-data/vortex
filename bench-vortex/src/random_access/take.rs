use std::iter;
use std::path::Path;

use arrow_array::types::Int64Type;
use arrow_array::{PrimitiveArray, RecordBatch};
use arrow_select::concat::concat_batches;
use arrow_select::take::take_record_batch;
use futures::stream;
use itertools::Itertools;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::arrow::async_reader::AsyncFileReader;
use parquet::file::metadata::RowGroupMetaData;
use stream::StreamExt;
use vortex::aliases::hash_map::HashMap;
use vortex::buffer::Buffer;
use vortex::error::VortexResult;
use vortex::file::VortexOpenOptions;
use vortex::io::{TokioFile, VortexReadAt};
use vortex::{Array, ArrayRef, IntoArray};

pub async fn take_vortex_tokio(path: &Path, indices: Buffer<u64>) -> VortexResult<ArrayRef> {
    take_vortex(TokioFile::open(path)?, indices).await
}

pub async fn take_parquet(path: &Path, indices: Buffer<u64>) -> VortexResult<RecordBatch> {
    let file = tokio::fs::File::open(path).await?;
    parquet_take_from_stream(file, indices).await
}

async fn take_vortex<T: VortexReadAt + Send>(
    reader: T,
    indices: Buffer<u64>,
) -> VortexResult<ArrayRef> {
    VortexOpenOptions::file()
        .open(reader)
        .await?
        .scan()?
        .with_row_indices(indices)
        .read_all()
        .await?
        // For equivalence.... we decompress to make sure we're not cheating too much.
        .to_canonical()
        .map(|a| a.into_array())
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
    let row_group_offsets = iter::once(0)
        .chain(
            builder
                .metadata()
                .row_groups()
                .iter()
                .map(RowGroupMetaData::num_rows),
        )
        .scan(0i64, |acc, x| {
            *acc += x;
            Some(*acc)
        })
        .collect::<Vec<_>>();

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
        .build()?;

    let schema = reader.schema().clone();

    let batches = reader
        .enumerate()
        .map(|(idx, batch)| {
            let batch = batch.unwrap();
            let indices = PrimitiveArray::<Int64Type>::from(row_group_indices[idx].clone());
            take_record_batch(&batch, &indices).unwrap()
        })
        .collect::<Vec<_>>()
        .await;

    Ok(concat_batches(&schema, &batches)?)
}
