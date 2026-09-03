// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::path::PathBuf;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::RecordBatch;
use arrow_ipc::writer::FileWriter;
use async_trait::async_trait;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use vortex::array::ArrayRef;

use crate::Format;
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
