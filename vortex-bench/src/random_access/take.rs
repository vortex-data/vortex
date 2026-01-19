// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::path::Path;
use std::path::PathBuf;

use arrow_array::PrimitiveArray;
use arrow_array::RecordBatch;
use arrow_array::types::Int64Type;
use arrow_select::concat::concat_batches;
use arrow_select::take::take_record_batch;
use async_trait::async_trait;
use futures::stream;
use itertools::Itertools;
use parquet::arrow::ParquetRecordBatchStreamBuilder;
use parquet::arrow::arrow_reader::ArrowReaderOptions;
use parquet::file::metadata::RowGroupMetaData;
use stream::StreamExt;
use vortex::array::Array;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::stream::ArrayStreamExt;
use vortex::buffer::Buffer;
use vortex::file::OpenOptionsSessionExt;
use vortex::utils::aliases::hash_map::HashMap;

use crate::Format;
use crate::SESSION;
use crate::random_access::RandomAccessor;

/// Random accessor for Vortex format files.
pub struct VortexRandomAccessor {
    path: PathBuf,
    name: String,
    format: Format,
}

impl VortexRandomAccessor {
    /// Create a new Vortex random accessor for the given file path.
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            name: "random-access/vortex-tokio-local-disk".to_string(),
            format: Format::OnDiskVortex,
        }
    }

    /// Create a new Vortex random accessor with a custom name.
    pub fn with_name(path: PathBuf, name: impl Into<String>) -> Self {
        Self {
            path,
            name: name.into(),
            format: Format::OnDiskVortex,
        }
    }

    /// Create a new Vortex random accessor for compact format.
    pub fn compact(path: PathBuf) -> Self {
        Self {
            path,
            name: "random-access/vortex-compact-tokio-local-disk".to_string(),
            format: Format::VortexCompact,
        }
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

    fn path(&self) -> &PathBuf {
        &self.path
    }

    async fn take(&self, indices: Vec<u64>) -> anyhow::Result<usize> {
        let result = take_vortex(&self.path, indices.into()).await?;
        Ok(result.len())
    }
}

/// Random accessor for Parquet format files.
pub struct ParquetRandomAccessor {
    path: PathBuf,
    name: String,
}

impl ParquetRandomAccessor {
    /// Create a new Parquet random accessor for the given file path.
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            name: "random-access/parquet-tokio-local-disk".to_string(),
        }
    }

    /// Create a new Parquet random accessor with a custom name.
    pub fn with_name(path: PathBuf, name: impl Into<String>) -> Self {
        Self {
            path,
            name: name.into(),
        }
    }
}

#[async_trait]
impl RandomAccessor for ParquetRandomAccessor {
    fn format(&self) -> Format {
        Format::Parquet
    }

    fn name(&self) -> &str {
        &self.name
    }

    fn path(&self) -> &PathBuf {
        &self.path
    }

    async fn take(&self, indices: Vec<u64>) -> anyhow::Result<usize> {
        let result = take_parquet(&self.path, indices).await?;
        Ok(result.num_rows())
    }
}

async fn take_vortex(reader: impl AsRef<Path>, indices: Buffer<u64>) -> anyhow::Result<ArrayRef> {
    let array = SESSION
        .open_options()
        .open_path(reader.as_ref())
        .await?
        .scan()?
        .with_row_indices(indices)
        .into_array_stream()?
        .read_all()
        .await?;

    // We canonicalize / decompress for equivalence to Arrow's `RecordBatch`es.
    let mut ctx = SESSION.create_execution_ctx();
    // TODO(joe): should we go to a vector.
    Ok(array.execute::<Canonical>(&mut ctx)?.into_array())
}

pub async fn take_parquet(path: &Path, indices: Vec<u64>) -> anyhow::Result<RecordBatch> {
    let file = tokio::fs::File::open(path).await?;

    let builder = ParquetRecordBatchStreamBuilder::new_with_options(
        file,
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
