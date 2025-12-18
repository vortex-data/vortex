// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;

use crate::Format;

pub mod take;

// Re-export implementations
pub use take::ParquetRandomAccessor;
pub use take::VortexRandomAccessor;

/// Trait for format-specific random access (take) operations.
///
/// Implementations handle reading specific rows by index from a data source.
/// Each implementation wraps a prepared file path and knows how to read from it.
#[async_trait]
pub trait RandomAccessor: Send + Sync {
    /// The format this accessor handles.
    fn format(&self) -> Format;

    /// A descriptive name for this accessor (used in benchmark output).
    fn name(&self) -> &str;

    /// The file path this accessor reads from.
    fn path(&self) -> &PathBuf;

    /// Take rows at the given indices, returning the number of rows read.
    async fn take(&self, indices: Vec<u64>) -> Result<usize>;
}
