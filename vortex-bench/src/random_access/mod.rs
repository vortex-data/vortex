// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use anyhow::Result;
use async_trait::async_trait;

use crate::Format;

pub mod take;

// Re-export implementations
pub use take::ParquetProjectingAccessor;
pub use take::ParquetRandomAccessor;
pub use take::VortexProjectingAccessor;
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

/// A field path for nested field access.
///
/// Represents a path to a nested field, e.g., `["payload", "ref"]` for `payload.ref`.
pub type FieldPath = Vec<String>;

/// Trait for format-specific random access with field projection.
///
/// Extends RandomAccessor to support selecting specific nested fields.
#[async_trait]
pub trait ProjectingRandomAccessor: Send + Sync {
    /// The format this accessor handles.
    fn format(&self) -> Format;

    /// A descriptive name for this accessor (used in benchmark output).
    fn name(&self) -> &str;

    /// The file path this accessor reads from.
    fn path(&self) -> &PathBuf;

    /// Take rows at the given indices with a projection to a nested field.
    ///
    /// The field_path specifies the path to a nested field, e.g., `["payload", "ref"]`.
    async fn take_projected(&self, indices: Vec<u64>, field_path: &FieldPath) -> Result<usize>;
}
