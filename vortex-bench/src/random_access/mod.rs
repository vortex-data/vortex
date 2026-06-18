// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use anyhow::Result;
use arrow_array::RecordBatch;
use async_trait::async_trait;
use vortex::array::ArrayRef;

use crate::Format;

pub mod take;

// Re-export implementations
pub use take::ParquetRandomAccessor;
pub use take::VortexRandomAccessor;

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
