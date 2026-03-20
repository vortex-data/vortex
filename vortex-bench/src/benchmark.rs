// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Core benchmark trait and types.

use std::fs;
use std::path::Path;
use std::path::PathBuf;

use arrow_schema::Schema;
use glob::Pattern;
use url::Url;

use crate::BenchmarkDataset;
use crate::Format;
use crate::IdempotentPath;
use crate::vortex_panic;

/// Specification for a table in a benchmark dataset.
#[derive(Debug, Clone)]
pub struct TableSpec {
    pub name: &'static str,
    pub schema: Option<Schema>,
}

impl TableSpec {
    pub fn new(name: &'static str, schema: Option<Schema>) -> Self {
        Self { name, schema }
    }
}

/// How to load the queries for a benchmark.
#[derive(Debug, Clone)]
pub enum QuerySource {
    /// Load from a single SQL file, split by semicolons (clickbench, polarsignals, statpopgen).
    SemicolonDelimited(PathBuf),
    /// Load from numbered SQL files in a directory (tpch: q1.sql..q22.sql, tpcds: 01.sql..99.sql).
    NumberedFiles {
        dir: PathBuf,
        start: usize,
        end: usize,
        /// Format string for the filename, e.g. "q{}.sql" or "{:02}.sql"
        format: NumberedFileFormat,
    },
    /// Inline query strings (fineweb, gharchive).
    Inline(Vec<&'static str>),
}

/// Format for numbered query files.
#[derive(Debug, Clone)]
pub enum NumberedFileFormat {
    /// `q{idx}.sql` — used by TPC-H (q1.sql, q2.sql, ...)
    Q,
    /// `{idx:02}.sql` — used by TPC-DS (01.sql, 02.sql, ...)
    ZeroPadded,
}

impl QuerySource {
    /// Create a `SemicolonDelimited` source from a file in the `vortex-bench` crate root.
    pub fn sql_file(name: &str) -> Self {
        QuerySource::SemicolonDelimited(Path::new(env!("CARGO_MANIFEST_DIR")).join(name))
    }

    /// Create a `NumberedFiles` source with `q{N}.sql` naming (TPC-H style).
    pub fn numbered_q(subdir: &str, start: usize, end: usize) -> Self {
        QuerySource::NumberedFiles {
            dir: Path::new(env!("CARGO_MANIFEST_DIR")).join(subdir),
            start,
            end,
            format: NumberedFileFormat::Q,
        }
    }

    /// Create a `NumberedFiles` source with `{NN:02}.sql` naming (TPC-DS style).
    pub fn numbered_zero_padded(subdir: &str, start: usize, end: usize) -> Self {
        QuerySource::NumberedFiles {
            dir: Path::new(env!("CARGO_MANIFEST_DIR")).join(subdir),
            start,
            end,
            format: NumberedFileFormat::ZeroPadded,
        }
    }

    /// Load all queries from this source.
    pub fn load(&self) -> anyhow::Result<Vec<(usize, String)>> {
        match self {
            QuerySource::SemicolonDelimited(path) => {
                let content = fs::read_to_string(path)?;
                Ok(content
                    .split(';')
                    .map(|s| s.trim())
                    .filter(|s| !s.is_empty())
                    .map(|s| s.to_string())
                    .enumerate()
                    .collect())
            }
            QuerySource::NumberedFiles {
                dir,
                start,
                end,
                format,
            } => Ok((*start..=*end)
                .map(|idx| {
                    let filename = match format {
                        NumberedFileFormat::Q => format!("q{idx}.sql"),
                        NumberedFileFormat::ZeroPadded => format!("{idx:02}.sql"),
                    };
                    let path = dir.join(filename);
                    let sql = fs::read_to_string(&path).unwrap_or_else(|_| {
                        vortex_panic!("cannot load query from {}", path.display())
                    });
                    (idx, sql)
                })
                .collect()),
            QuerySource::Inline(queries) => {
                Ok(queries.iter().map(|s| s.to_string()).enumerate().collect())
            }
        }
    }
}

/// Controls the glob pattern used to find data files for a table.
#[derive(Debug, Clone)]
pub enum FilePattern {
    /// No pattern override — returns `None` from `pattern()`.
    Default,
    /// `{table_name}_*.{ext}` — used by TPC-H for multi-file partitioned tables.
    TablePrefix,
    /// `{table_name}.{ext}` — used by TPC-DS, PublicBI for single-file tables.
    TableExact,
    /// A fixed filename (ignoring table_name). e.g. `"stacktraces"` → `stacktraces.{ext}`.
    Fixed(&'static str),
}

impl FilePattern {
    /// Resolve this pattern for a given table name and format.
    #[expect(clippy::unwrap_in_result)]
    pub fn resolve(&self, table_name: &str, format: Format) -> Option<Pattern> {
        match self {
            FilePattern::Default => None,
            FilePattern::TablePrefix => Some(
                format!("{}_*.{}", table_name, format.ext())
                    .parse()
                    .expect("valid glob pattern"),
            ),
            FilePattern::TableExact => Some(
                format!("{}.{}", table_name, format.ext())
                    .parse()
                    .expect("valid glob pattern"),
            ),
            FilePattern::Fixed(name) => Some(
                format!("{}.{}", name, format.ext())
                    .parse()
                    .expect("valid glob pattern"),
            ),
        }
    }
}

/// All the metadata that describes a benchmark. Store this as a field on your
/// benchmark struct and implement `fn descriptor(&self) -> &BenchmarkDescriptor`
/// to get default implementations for every trait method except `generate_base_data`.
pub struct BenchmarkDescriptor {
    pub name: &'static str,
    pub display: String,
    pub data_url: Url,
    pub tables: Vec<TableSpec>,
    pub queries: QuerySource,
    pub expected_row_counts: Option<Vec<usize>>,
    pub dataset: BenchmarkDataset,
    pub file_pattern: FilePattern,
}

impl BenchmarkDescriptor {
    /// Start building a descriptor with required fields.
    pub fn new(name: &'static str, data_url: Url, dataset: BenchmarkDataset) -> Self {
        let display = dataset.to_string();
        Self {
            name,
            display,
            data_url,
            tables: Vec::new(),
            queries: QuerySource::Inline(Vec::new()),
            expected_row_counts: None,
            dataset,
            file_pattern: FilePattern::Default,
        }
    }

    pub fn with_display(mut self, display: String) -> Self {
        self.display = display;
        self
    }

    pub fn with_table(mut self, name: &'static str, schema: Option<Schema>) -> Self {
        self.tables.push(TableSpec::new(name, schema));
        self
    }

    pub fn with_tables(mut self, tables: Vec<TableSpec>) -> Self {
        self.tables = tables;
        self
    }

    pub fn with_queries(mut self, queries: QuerySource) -> Self {
        self.queries = queries;
        self
    }

    pub fn with_expected_row_counts(mut self, counts: Vec<usize>) -> Self {
        self.expected_row_counts = Some(counts);
        self
    }

    pub fn with_file_pattern(mut self, pattern: FilePattern) -> Self {
        self.file_pattern = pattern;
        self
    }
}

/// Resolve a data URL for a benchmark, handling both local and remote cases.
///
/// For local: creates a `file://` URL pointing to `data/{local_subpath}/`.
/// For remote: parses the URL and ensures it ends with `/`.
pub fn resolve_data_url(local_subpath: &str, remote_data_dir: Option<&str>) -> anyhow::Result<Url> {
    match remote_data_dir {
        None => {
            let path = local_subpath.to_data_path();
            Url::from_directory_path(&path).map_err(|_| {
                anyhow::anyhow!("Failed to create URL from directory path: {:?}", path)
            })
        }
        Some(remote) => {
            if !remote.ends_with('/') {
                tracing::warn!(
                    "Supply a remote data dir ending in a slash, e.g. s3://bucket/path/"
                );
            }
            tracing::info!(
                "Assuming data already exists at remote URL: {}. \
                 If it does not, run without --use-remote-data-dir first to generate locally.",
                remote,
            );
            let mut url = Url::parse(remote)?;
            if !url.path().ends_with('/') {
                url.set_path(&format!("{}/", url.path()));
            }
            Ok(url)
        }
    }
}

/// Core trait for benchmark datasets.
///
/// Implementations provide queries, data generation, and metadata for running
/// benchmarks across different engines and formats.
///
/// **To reduce boilerplate**, store a [`BenchmarkDescriptor`] on your struct and
/// implement [`descriptor()`](Benchmark::descriptor). All other methods have
/// default implementations that delegate to the descriptor. You only need to
/// implement [`generate_base_data()`](Benchmark::generate_base_data) yourself.
#[async_trait::async_trait]
pub trait Benchmark: Send + Sync {
    /// Return the descriptor for this benchmark.
    fn descriptor(&self) -> &BenchmarkDescriptor;

    /// Generate or prepare base data for the benchmark (typically Parquet format).
    /// This is the canonical source data that can be converted to other formats.
    /// This should be idempotent — safe to call multiple times.
    async fn generate_base_data(&self) -> anyhow::Result<()>;

    /// Get all available queries for this benchmark.
    fn queries(&self) -> anyhow::Result<Vec<(usize, String)>> {
        self.descriptor().queries.load()
    }

    /// Get expected row counts for validation (optional).
    /// If None, no validation will be performed.
    fn expected_row_counts(&self) -> Option<Vec<usize>> {
        self.descriptor().expected_row_counts.clone()
    }

    fn dataset(&self) -> BenchmarkDataset {
        self.descriptor().dataset.clone()
    }

    /// Get the name of the benchmark dataset.
    fn dataset_name(&self) -> &str {
        self.descriptor().name
    }

    /// Get the table names for this dataset.
    fn tables(&self) -> Vec<&'static str> {
        self.table_specs().iter().map(|ts| ts.name).collect()
    }

    /// Format a path for the given format and base URL.
    fn format_path(&self, format: Format, base_url: &Url) -> anyhow::Result<Url> {
        Ok(base_url.join(&format!("{}/", format.name()))?)
    }

    /// Get display string for the dataset (used in measurements).
    fn dataset_display(&self) -> String {
        self.descriptor().display.clone()
    }

    fn data_url(&self) -> &Url {
        &self.descriptor().data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        self.descriptor().tables.clone()
    }

    fn pattern(&self, table_name: &str, format: Format) -> Option<Pattern> {
        self.descriptor().file_pattern.resolve(table_name, format)
    }
}
