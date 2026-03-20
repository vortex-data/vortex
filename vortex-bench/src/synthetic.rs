// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generic framework for synthetic SQL benchmarks.
//!
//! Define tables (columns + generators) and queries in one place, and the
//! framework handles Parquet generation and the [`Benchmark`] trait impl.
//!
//! # Example
//!
//! ```ignore
//! use std::sync::Arc;
//! use arrow_array::{Float64Array, Int64Array};
//! use arrow_schema::DataType;
//!
//! let benchmark = SyntheticBenchmark::builder("my-bench", 1_000_000)
//!     .table("readings", |t| {
//!         t.column("id", DataType::Int64, |start, len, _rng| {
//!             Arc::new(Int64Array::from_iter_values(
//!                 (start as i64)..((start + len) as i64),
//!             ))
//!         });
//!         t.column("value", DataType::Float64, |start, len, rng| {
//!             Arc::new(Float64Array::from_iter_values(
//!                 (0..len).map(|j| 42.0 + rng.random::<f64>()),
//!             ))
//!         });
//!     })
//!     .queries(&[
//!         "SELECT SUM(value) FROM readings",
//!         "SELECT id, value FROM readings WHERE value > 42.5",
//!     ])
//!     .build()?;
//! ```

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use arrow_array::ArrayRef;
use arrow_array::RecordBatch;
use arrow_schema::DataType;
use arrow_schema::Field;
use arrow_schema::Schema;
use parquet::arrow::ArrowWriter;
use rand::rngs::StdRng;
use rand::SeedableRng;
use url::Url;

use crate::benchmark::TableSpec;
use crate::datasets::BenchmarkDataset;
use crate::Benchmark;
use crate::Format;
use crate::IdempotentPath;

/// Function that generates one column's worth of Arrow data for a batch.
///
/// Arguments: `(batch_start, batch_len, rng)`.
pub type ColumnGeneratorFn =
    Arc<dyn Fn(usize, usize, &mut StdRng) -> ArrayRef + Send + Sync>;

/// A column in a synthetic table.
pub struct SyntheticColumn {
    /// Column name.
    pub name: &'static str,
    /// Arrow data type.
    pub data_type: DataType,
    /// Whether the column is nullable.
    pub nullable: bool,
    /// Generator function producing an [`ArrayRef`] for each batch.
    pub generator: ColumnGeneratorFn,
}

/// A table in a synthetic benchmark.
pub struct SyntheticTable {
    /// Table name (used in SQL and as the Parquet filename).
    pub name: &'static str,
    /// Column definitions.
    pub columns: Vec<SyntheticColumn>,
}

impl SyntheticTable {
    /// Derive the Arrow schema from column definitions.
    pub fn schema(&self) -> Schema {
        Schema::new(
            self.columns
                .iter()
                .map(|c| Field::new(c.name, c.data_type.clone(), c.nullable))
                .collect::<Vec<_>>(),
        )
    }
}

/// Builder for defining a [`SyntheticTable`] via closures.
pub struct SyntheticTableBuilder {
    name: &'static str,
    columns: Vec<SyntheticColumn>,
}

impl SyntheticTableBuilder {
    fn new(name: &'static str) -> Self {
        Self {
            name,
            columns: Vec::new(),
        }
    }

    /// Add a non-nullable column with a generator function.
    pub fn column(
        &mut self,
        name: &'static str,
        data_type: DataType,
        generator: impl Fn(usize, usize, &mut StdRng) -> ArrayRef + Send + Sync + 'static,
    ) -> &mut Self {
        self.columns.push(SyntheticColumn {
            name,
            data_type,
            nullable: false,
            generator: Arc::new(generator),
        });
        self
    }

    /// Add a nullable column with a generator function.
    pub fn nullable_column(
        &mut self,
        name: &'static str,
        data_type: DataType,
        generator: impl Fn(usize, usize, &mut StdRng) -> ArrayRef + Send + Sync + 'static,
    ) -> &mut Self {
        self.columns.push(SyntheticColumn {
            name,
            data_type,
            nullable: true,
            generator: Arc::new(generator),
        });
        self
    }

    fn build(self) -> SyntheticTable {
        SyntheticTable {
            name: self.name,
            columns: self.columns,
        }
    }
}

/// Builder for constructing a [`SyntheticBenchmark`].
pub struct SyntheticBenchmarkBuilder {
    name: String,
    n_rows: usize,
    tables: Vec<SyntheticTable>,
    queries: Vec<String>,
}

impl SyntheticBenchmarkBuilder {
    /// Add a table defined via a builder closure.
    pub fn table(
        mut self,
        name: &'static str,
        define: impl FnOnce(&mut SyntheticTableBuilder),
    ) -> Self {
        let mut builder = SyntheticTableBuilder::new(name);
        define(&mut builder);
        self.tables.push(builder.build());
        self
    }

    /// Set the SQL queries for this benchmark.
    pub fn queries(mut self, queries: &[&str]) -> Self {
        self.queries = queries.iter().map(|q| (*q).to_string()).collect();
        self
    }

    /// Build the benchmark.
    pub fn build(self) -> Result<SyntheticBenchmark> {
        SyntheticBenchmark::from_parts(self.name, self.n_rows, self.tables, self.queries)
    }
}

/// Batch size for writing Parquet.
const BATCH_SIZE: usize = 100_000;

/// A generic synthetic SQL benchmark.
///
/// Tables, columns, data generators, and queries are all defined together in
/// one place. The framework handles Parquet generation and implements the
/// [`Benchmark`] trait so it plugs directly into the DataFusion and DuckDB
/// benchmark runners.
pub struct SyntheticBenchmark {
    name: String,
    tables: Vec<SyntheticTable>,
    queries: Vec<String>,
    n_rows: usize,
    data_url: Url,
}

impl SyntheticBenchmark {
    /// Start building a new synthetic benchmark.
    pub fn builder(name: impl Into<String>, n_rows: usize) -> SyntheticBenchmarkBuilder {
        SyntheticBenchmarkBuilder {
            name: name.into(),
            n_rows,
            tables: Vec::new(),
            queries: Vec::new(),
        }
    }

    fn from_parts(
        name: String,
        n_rows: usize,
        tables: Vec<SyntheticTable>,
        queries: Vec<String>,
    ) -> Result<Self> {
        let data_path = name.to_data_path().join(format!("{n_rows}/"));
        let data_url =
            Url::from_directory_path(data_path).map_err(|_| anyhow::anyhow!("bad data path"))?;

        Ok(Self {
            name,
            tables,
            queries,
            n_rows,
            data_url,
        })
    }

    fn parquet_dir(&self) -> Result<std::path::PathBuf> {
        self.data_url
            .join("parquet/")?
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("failed to convert data URL to filesystem path"))
    }
}

/// Generate a single Parquet file for one synthetic table.
fn generate_table_parquet(
    table: &SyntheticTable,
    n_rows: usize,
    path: &Path,
) -> Result<()> {
    let schema = Arc::new(table.schema());
    let file = File::create(path)?;
    let mut writer = ArrowWriter::try_new(file, schema.clone(), None)?;
    let mut rng = StdRng::seed_from_u64(42);

    for batch_start in (0..n_rows).step_by(BATCH_SIZE) {
        let batch_len = BATCH_SIZE.min(n_rows - batch_start);

        let columns: Vec<ArrayRef> = table
            .columns
            .iter()
            .map(|col| (col.generator)(batch_start, batch_len, &mut rng))
            .collect();

        let batch = RecordBatch::try_new(schema.clone(), columns)?;
        writer.write(&batch)?;
    }

    writer.close()?;
    Ok(())
}

#[async_trait::async_trait]
impl Benchmark for SyntheticBenchmark {
    fn queries(&self) -> Result<Vec<(usize, String)>> {
        Ok(self
            .queries
            .iter()
            .cloned()
            .enumerate()
            .collect())
    }

    async fn generate_base_data(&self) -> Result<()> {
        if self.data_url.scheme() != "file" {
            anyhow::bail!(
                "unsupported URL scheme '{}' - only 'file://' URLs are supported",
                self.data_url.scheme()
            );
        }

        let parquet_dir = self.parquet_dir()?;

        for table in &self.tables {
            let parquet_path = parquet_dir.join(format!("{}.parquet", table.name));
            crate::idempotent(&parquet_path, |tmp_path| {
                tracing::info!(
                    n_rows = self.n_rows,
                    table = table.name,
                    path = %parquet_path.display(),
                    "generating synthetic benchmark parquet data"
                );
                generate_table_parquet(table, self.n_rows, tmp_path)
            })?;
        }
        Ok(())
    }

    fn dataset(&self) -> BenchmarkDataset {
        BenchmarkDataset::Synthetic {
            name: self.name.clone(),
            n_rows: self.n_rows,
        }
    }

    fn dataset_name(&self) -> &str {
        &self.name
    }

    fn dataset_display(&self) -> String {
        format!("{}(n_rows={})", self.name, self.n_rows)
    }

    fn data_url(&self) -> &Url {
        &self.data_url
    }

    fn table_specs(&self) -> Vec<TableSpec> {
        self.tables
            .iter()
            .map(|t| TableSpec::new(t.name, Some(t.schema())))
            .collect()
    }

    #[expect(clippy::expect_used, clippy::unwrap_in_result)]
    fn pattern(&self, table_name: &str, format: Format) -> Option<glob::Pattern> {
        Some(
            format!("{table_name}.{}", format.ext())
                .parse()
                .expect("valid glob pattern"),
        )
    }
}
