// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core benchmark trait and types.

use arrow_schema::Schema;
use glob::Pattern;
use url::Url;

use crate::BenchmarkDataset;
use crate::Engine;
use crate::Format;

/// Specification for a table in a benchmark dataset.
#[derive(Debug)]
pub struct TableSpec {
    pub name: &'static str,
    pub schema: Option<Schema>,
}

impl TableSpec {
    pub fn new(name: &'static str, schema: Option<Schema>) -> Self {
        Self { name, schema }
    }
}

/// Core trait for benchmark datasets.
///
/// Implementations provide queries, data generation, and metadata for running
/// benchmarks across different engines and formats.
#[async_trait::async_trait]
pub trait Benchmark: Send + Sync {
    /// Get all available queries for this benchmark
    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>>;

    /// SQL an `engine` must run before this benchmark's queries (e.g. loading engine
    /// extensions). Runners replay these after every (re)open. Default: none.
    fn engine_init_sql(&self, _engine: Engine) -> Vec<String> {
        Vec::new()
    }

    /// Generate or prepare base data for the benchmark (typically Parquet format).
    /// This is the canonical source data that can be converted to other formats.
    /// This should be idempotent - safe to call multiple times.
    ///
    /// The `data-gen` binary calls this method before creating requested derived formats.
    /// Engine-specific benchmark binaries should only run against prepared data.
    async fn generate_base_data(&self) -> anyhow::Result<()>;

    /// Get expected row counts for validation (optional)
    /// If None, no validation will be performed
    fn expected_row_counts(&self) -> Option<Vec<usize>> {
        None
    }

    fn dataset(&self) -> BenchmarkDataset;

    /// Get the name of the benchmark dataset
    fn dataset_name(&self) -> &str;

    /// Get the table names for this dataset (used for TPC benchmarks)
    fn tables(&self) -> Vec<&'static str> {
        self.table_specs().iter().map(|ts| ts.name).collect()
    }

    /// Format a path for the given format and base URL
    fn format_path(&self, format: Format, base_url: &Url) -> anyhow::Result<Url> {
        Ok(base_url.join(&format!("{}/", format.name()))?)
    }

    /// Get display string for the dataset (used in measurements)
    fn dataset_display(&self) -> String;

    fn data_url(&self) -> &Url;

    fn table_specs(&self) -> Vec<TableSpec>;

    fn pattern(&self, table_name: &str, format: Format) -> Option<Pattern> {
        _ = table_name;
        _ = format;
        None
    }
}
