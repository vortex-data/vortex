// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter::once;
use std::path::PathBuf;

use anyhow::anyhow;
use arrow_array::PrimitiveArray;
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
use tokio::fs::File;
use vortex::array::Array;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::stream::ArrayStreamExt;
use vortex::buffer::Buffer;
use vortex::file::OpenOptionsSessionExt;
use vortex::file::VortexFile;
use vortex::utils::aliases::hash_map::HashMap;

use crate::Format;
use crate::SESSION;
use crate::random_access::RandomAccessor;

/// Random accessor for Vortex format files.
///
/// After `open()`, the file handle is stored and reused across `take()` calls.
pub struct VortexRandomAccessor {
    path: PathBuf,
    name: String,
    format: Format,
    file: Option<VortexFile>,
}

impl VortexRandomAccessor {
    /// Create a new Vortex random accessor for the given file path.
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            name: "random-access/vortex-tokio-local-disk".to_string(),
            format: Format::OnDiskVortex,
            file: None,
        }
    }

    /// Create a new Vortex random accessor with a custom name.
    pub fn with_name(path: PathBuf, name: impl Into<String>) -> Self {
        Self {
            path,
            name: name.into(),
            format: Format::OnDiskVortex,
            file: None,
        }
    }

    /// Create a new Vortex random accessor with a custom name and format.
    pub fn with_name_and_format(path: PathBuf, name: impl Into<String>, format: Format) -> Self {
        Self {
            path,
            name: name.into(),
            format,
            file: None,
        }
    }

    /// Create a new Vortex random accessor for compact format.
    pub fn compact(path: PathBuf) -> Self {
        Self {
            path,
            name: "random-access/vortex-compact-tokio-local-disk".to_string(),
            format: Format::VortexCompact,
            file: None,
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

    async fn open(&mut self) -> anyhow::Result<()> {
        let file = SESSION.open_options().open_path(&self.path).await?;
        self.file = Some(file);
        Ok(())
    }

    async fn take(&self, indices: &[u64]) -> anyhow::Result<usize> {
        let file = self
            .file
            .as_ref()
            .ok_or_else(|| anyhow!("accessor not opened; call open() first"))?;

        let indices_buf: Buffer<u64> = Buffer::from(indices.to_vec());
        let array = file
            .scan()?
            .with_row_indices(indices_buf)
            .into_array_stream()?
            .read_all()
            .await?;

        // We canonicalize / decompress for equivalence to Arrow's `RecordBatch`es.
        let mut ctx = SESSION.create_execution_ctx();
        let canonical = array.execute::<Canonical>(&mut ctx)?.into_array();
        Ok(canonical.len())
    }
}

/// Pre-computed Parquet metadata stored after `open()`.
struct ParquetMetadata {
    /// Cumulative row offsets per row group (length = num_row_groups + 1).
    row_group_offsets: Vec<i64>,
    /// Path to the Parquet file (for re-opening on each take).
    path: PathBuf,
}

/// Random accessor for Parquet format files.
///
/// After `open()`, the file metadata and row group offsets are stored and
/// reused to map indices to row groups in each `take()` call.
pub struct ParquetRandomAccessor {
    path: PathBuf,
    name: String,
    metadata: Option<ParquetMetadata>,
}

impl ParquetRandomAccessor {
    /// Create a new Parquet random accessor for the given file path.
    pub fn new(path: PathBuf) -> Self {
        Self {
            path,
            name: "random-access/parquet-tokio-local-disk".to_string(),
            metadata: None,
        }
    }

    /// Create a new Parquet random accessor with a custom name.
    pub fn with_name(path: PathBuf, name: impl Into<String>) -> Self {
        Self {
            path,
            name: name.into(),
            metadata: None,
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

    async fn open(&mut self) -> anyhow::Result<()> {
        let file = File::open(&self.path).await?;
        let builder = ParquetRecordBatchStreamBuilder::new_with_options(
            file,
            ArrowReaderOptions::new().with_page_index(true),
        )
        .await?;

        let row_group_offsets = once(0)
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

        self.metadata = Some(ParquetMetadata {
            row_group_offsets,
            path: self.path.clone(),
        });

        Ok(())
    }

    async fn take(&self, indices: &[u64]) -> anyhow::Result<usize> {
        let meta = self
            .metadata
            .as_ref()
            .ok_or_else(|| anyhow!("accessor not opened; call open() first"))?;

        // Map indices to row groups.
        let mut row_groups = HashMap::new();
        for &idx in indices {
            let row_group_idx = meta
                .row_group_offsets
                .binary_search(&(idx as i64))
                .unwrap_or_else(|e| e - 1);
            row_groups
                .entry(row_group_idx)
                .or_insert_with(Vec::new)
                .push((idx as i64) - meta.row_group_offsets[row_group_idx]);
        }

        let sorted_row_group_keys = row_groups.keys().copied().sorted().collect_vec();
        let row_group_indices = sorted_row_group_keys
            .iter()
            .map(|i| row_groups[i].clone())
            .collect_vec();

        // Re-open the file for reading (Parquet builder consumes the file handle).
        let file = File::open(&meta.path).await?;
        let builder = ParquetRecordBatchStreamBuilder::new_with_options(
            file,
            ArrowReaderOptions::new().with_page_index(true),
        )
        .await?;

        let reader = builder
            .with_row_groups(sorted_row_group_keys)
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

        let result = concat_batches(&schema, &batches)?;
        Ok(result.num_rows())
    }
}
