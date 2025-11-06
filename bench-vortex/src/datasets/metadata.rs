// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Dataset metadata trait for simplified dataset model
//!
//! This module provides a trait-based approach to dataset metadata,
//! allowing benchmarks to define their dataset structure without
//! coupling to specific implementations.

use std::fmt::Display;

/// Information about a single table in a dataset
#[derive(Debug, Clone)]
pub struct TableInfo {
    /// Name of the table
    pub name: String,
    /// File pattern for matching files (e.g., "lineitem*.parquet")
    pub file_pattern: String,
    /// Optional schema hint
    pub schema_hint: Option<arrow_schema::Schema>,
}

impl TableInfo {
    pub fn new(name: impl Into<String>, pattern: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            file_pattern: pattern.into(),
            schema_hint: None,
        }
    }

    pub fn with_schema(mut self, schema: arrow_schema::Schema) -> Self {
        self.schema_hint = Some(schema);
        self
    }
}

/// Trait for dataset metadata
///
/// This trait provides the minimal interface needed for benchmarks
/// to describe their dataset structure. The actual behavior
/// (registration, conversion, etc.) is handled by the benchmark
/// implementations rather than the dataset itself.
pub trait DatasetMetadata: Display + Send + Sync {
    /// Get the dataset name
    fn name(&self) -> &str;

    /// Get the list of tables in this dataset
    fn tables(&self) -> Vec<TableInfo>;

    /// Get the dataset variant identifier (e.g., scale factor, flavor)
    ///
    /// This is used for constructing paths and display names.
    fn variant(&self) -> String {
        String::new()
    }

    /// Get the display name for this dataset
    ///
    /// Default implementation combines name and variant.
    fn display_name(&self) -> String {
        if self.variant().is_empty() {
            self.name().to_string()
        } else {
            format!("{}_{}", self.name(), self.variant())
        }
    }
}