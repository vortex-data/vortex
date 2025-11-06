// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Format conversion infrastructure for benchmarks
//!
//! This module provides a trait-based system for converting between different
//! data formats (Parquet, Vortex, Lance, etc.) in a consistent way.

use std::path::Path;

use anyhow::Result;
use async_trait::async_trait;

use crate::Format;

pub mod converters;
pub mod registry;

pub use registry::{ConverterRegistry, global_registry};

/// Strategy for compacting data during conversion
///
/// Note: This duplicates the one in lib.rs but provides clearer semantics
/// for the conversion module. In future, we should consolidate these.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactionStrategy {
    /// Use default compaction settings
    Default,
    /// Use aggressive compaction for smaller size
    Compact,
    /// No compaction
    None,
}

/// Options for format conversion
#[derive(Debug, Clone)]
pub struct ConversionOptions {
    /// Strategy for compaction (Vortex formats only)
    pub compaction: CompactionStrategy,
    /// Number of concurrent tasks for parallel conversion
    pub parallelism: Option<usize>,
    /// Whether to overwrite existing files
    pub overwrite: bool,
}

impl Default for ConversionOptions {
    fn default() -> Self {
        Self {
            compaction: CompactionStrategy::Default,
            parallelism: None,
            overwrite: false,
        }
    }
}

/// Trait for converting data between formats
///
/// Implementations should be idempotent - running the same conversion
/// twice should not re-process files that already exist (unless overwrite is true).
#[async_trait]
pub trait FormatConverter: Send + Sync {
    /// Convert data from source format to target format
    ///
    /// # Arguments
    /// * `source_path` - Path to source data files
    /// * `target_path` - Path where converted files should be written
    /// * `options` - Conversion options
    async fn convert(
        &self,
        source_path: &Path,
        target_path: &Path,
        options: &ConversionOptions,
    ) -> Result<()>;

    /// Check if this converter supports the given format pair
    fn supports(&self, source_format: Format, target_format: Format) -> bool;

    /// Get a human-readable name for this converter
    fn name(&self) -> &str;

    /// Get the source format this converter expects
    fn source_format(&self) -> Format;

    /// Get the target format this converter produces
    fn target_format(&self) -> Format;
}

/// Result of a conversion operation
#[derive(Debug)]
pub struct ConversionResult {
    /// Number of files converted
    pub files_converted: usize,
    /// Number of files skipped (already existed)
    pub files_skipped: usize,
    /// Total bytes written
    pub bytes_written: u64,
    /// Total time taken
    pub duration: std::time::Duration,
}

/// Helper function to convert data between formats using the global registry
///
/// This is a convenience function that looks up the appropriate converter
/// and executes it with the given options.
pub async fn convert_format(
    source_path: &Path,
    target_path: &Path,
    source_format: Format,
    target_format: Format,
    options: ConversionOptions,
) -> Result<()> {
    use anyhow::Context;

    let registry = global_registry();

    let converter = registry
        .find_converter(source_format, target_format)
        .with_context(|| {
            format!(
                "No converter available for {} -> {}. Available converters:\n{}",
                source_format,
                target_format,
                registry.list_converters().join("\n")
            )
        })?;

    converter.convert(source_path, target_path, &options).await
}