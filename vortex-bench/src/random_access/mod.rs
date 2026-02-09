// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use anyhow::Result;
use async_trait::async_trait;

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

/// Trait for a benchmark dataset that knows how to write files and create accessors.
#[async_trait]
pub trait BenchDataset: Send + Sync {
    /// A descriptive name for this dataset (used in benchmark output and CLI).
    fn name(&self) -> &str;

    /// The total number of rows in this dataset.
    fn row_count(&self) -> u64;

    /// Create a format-specific random accessor for this dataset.
    ///
    /// This prepares the data file (writing it if necessary) and returns an
    /// accessor that can be opened and used for random access benchmarks.
    async fn create(&self, format: Format) -> Result<Box<dyn RandomAccessor>>;
}

/// Trait for format-specific random access (take) operations.
///
/// Implementations handle reading specific rows by index from a data source.
/// The lifecycle is: construct -> `open()` (parse metadata) -> `take()` (I/O).
#[async_trait]
pub trait RandomAccessor: Send + Sync {
    /// A descriptive name for this accessor (used in benchmark output).
    fn name(&self) -> &str;

    /// The format this accessor handles.
    fn format(&self) -> Format;

    /// Open the file and parse metadata. This is not timed in benchmarks.
    async fn open(&mut self) -> Result<()>;

    /// Take rows at the given indices, returning the number of rows read.
    async fn take(&self, indices: &[u64]) -> Result<usize>;
}
