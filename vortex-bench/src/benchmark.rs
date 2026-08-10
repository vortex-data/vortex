// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core benchmark trait and types.

use std::path::Path;

use arrow_schema::Schema;
use glob::Pattern;
use url::Url;

use crate::BenchmarkDataset;
use crate::Engine;
use crate::Format;
use crate::setup::SetupCtx;

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

    /// Formats this benchmark can materialize without help.
    ///
    /// [`prepare_data`] derives the rest by converting between Parquet and Vortex, in either
    /// direction. Listing only Parquet is the common case; suites that write Vortex straight
    /// from a row generator list it too, so requesting Vortex skips a needless round trip
    /// through Parquet on disk.
    ///
    /// [`prepare_data`]: crate::setup::prepare_data
    fn native_formats(&self) -> &[Format] {
        &[Format::Parquet]
    }

    /// Materialize `format` into `ctx.staging()`, registering each produced file with
    /// [`SetupCtx::emit`].
    ///
    /// Only called for formats listed in [`Benchmark::native_formats`]. Must be idempotent:
    /// the staging directory persists across runs, so work already done should be skipped
    /// rather than repeated.
    async fn setup(&self, ctx: &SetupCtx, format: Format) -> anyhow::Result<()>;

    /// Prepare benchmark- and format-specific data beyond what [`Benchmark::setup`] or the
    /// Parquet-to-Vortex conversion produced. Called once per requested format, after that
    /// format's data exists. Default: nothing.
    async fn prepare_format(&self, _format: Format, _base_path: &Path) -> anyhow::Result<()> {
        Ok(())
    }

    /// Get expected row counts for validation (optional)
    /// If None, no validation will be performed
    fn expected_row_counts(&self) -> Option<Vec<usize>> {
        None
    }

    fn dataset(&self) -> BenchmarkDataset;

    /// Repo-relative path of the markdown explainer for this benchmark suite, linked from the
    /// title of CI benchmark PR comments. Required: every suite must ship a doc.
    fn doc_path(&self) -> &'static str;

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
