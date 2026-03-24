// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DuckDB context for benchmarks.

use std::ops::Deref;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use anyhow::Result;
use tracing::trace;
use vortex::error::VortexExpect;
use vortex_bench::Benchmark;
use vortex_bench::Format;
use vortex_bench::IdempotentPath;
use vortex_bench::generate_duckdb_registration_sql;
use vortex_bench::runner::BenchmarkQueryResult;
use vortex_duckdb::duckdb::Config;
use vortex_duckdb::duckdb::Connection;
use vortex_duckdb::duckdb::Database;
use vortex_duckdb::duckdb::QueryResult;

/// DuckDB context for benchmarks.
pub struct DuckClient {
    db: Option<Database>,
    connection: Option<Connection>,
    pub db_path: PathBuf,
    pub threads: Option<usize>,
}

impl DuckClient {
    pub fn connection(&self) -> &Connection {
        self.connection
            .as_ref()
            .vortex_expect("DuckClient connection accessed after close")
    }

    /// Create a new DuckDB context with a database at `{data_url}/{format}/duckdb.db`.
    pub fn new(
        benchmark: &dyn Benchmark,
        format: Format,
        delete_database: bool,
        threads: Option<usize>,
    ) -> Result<Self> {
        let data_url = benchmark.data_url();
        let base_path = if data_url.scheme() == "file" {
            data_url
                .to_file_path()
                .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", data_url))?
        } else {
            format!("{name}/{format}/", name = benchmark.dataset_name()).to_data_path()
        };
        let dir = base_path.join(format.name());
        std::fs::create_dir_all(&dir)?;
        let db_path = dir.join("duckdb.db");

        tracing::info!(db_path = %db_path.display(), "Opening DuckDB");

        if delete_database && db_path.exists() {
            std::fs::remove_file(&db_path)?;
        }

        let (db, connection) = Self::open_and_setup_database(Some(db_path.clone()), threads)?;

        Ok(Self {
            db: Some(db),
            connection: Some(connection),
            db_path,
            threads,
        })
    }

    pub fn open_and_setup_database(
        path: Option<PathBuf>,
        threads: Option<usize>,
    ) -> Result<(Database, Connection)> {
        let mut config = Config::new().vortex_expect("failed to create duckdb config");

        // Set DuckDB thread count if specified
        if let Some(thread_count) = threads {
            config.set("threads", &format!("{}", thread_count))?;
        }

        let db = match path {
            Some(path) => Database::open_with_config(path, config),
            None => Database::open_in_memory_with_config(config),
        }?;

        let connection = db.connect()?;
        vortex_duckdb::initialize(&db)?;

        // Enable Parquet metadata cache for all benchmark runs.
        //
        // `parquet_metadata_cache` is an extension-specific option that's
        // only available after the Parquet extension is loaded. The Parquet
        // extension is loaded after the connection is established.
        //
        // Passing the option to `open_with_config` before leads to
        // "Invalid Input Error: The following options were not recognized:
        // parquet_metadata_cache" when running DuckDB in debug mode.
        connection.query("SET parquet_metadata_cache = true")?;

        Ok((db, connection))
    }

    pub fn reopen(&mut self) -> Result<()> {
        // Close the old database before opening a new one on the same file path.
        // DuckDB cannot have two instances on the same file simultaneously — the new
        // instance may read an inconsistent state, causing deserialization
        // errors. Drop connection before database (connection depends on database).
        self.connection.take();
        self.db.take();

        let (db, connection) =
            Self::open_and_setup_database(Some(self.db_path.clone()), self.threads)?;

        self.db = Some(db);
        self.connection = Some(connection);

        Ok(())
    }

    pub fn new_in_memory() -> Result<Self> {
        let dir = std::env::temp_dir()
            .join("vortex-duckdb-bench")
            .join("in-memory");
        std::fs::create_dir_all(&dir)?;
        let db_path = dir.join("duckdb.db");
        let (db, connection) = Self::open_and_setup_database(Some(db_path.clone()), None)?;
        Ok(Self {
            db: Some(db),
            connection: Some(connection),
            db_path,
            threads: None,
        })
    }

    /// Execute DuckDB queries for benchmarks using the internal connection.
    /// Returns `(row_count, optional_timing)` where `optional_timing` is the query's
    /// internal execution time if available.
    pub fn execute_query(&self, query: &str) -> Result<(usize, Option<Duration>)> {
        trace!("execute duckdb query: {query}");
        let time_instant = Instant::now();
        let result = self.connection().query(query)?;
        let query_time = time_instant.elapsed();

        let row_count = usize::try_from(result.row_count()).vortex_expect("row count overflow");

        // TODO: Extract DuckDB's internal timing from profiling info if available
        Ok((row_count, Some(query_time)))
    }

    /// Register tables for benchmarks using the internal connection.
    pub fn register_tables<B: Benchmark + ?Sized>(
        &self,
        benchmark: &B,
        file_format: Format,
    ) -> Result<()> {
        let object_type = match file_format {
            Format::Parquet | Format::OnDiskVortex | Format::VortexCompact => "VIEW",
            Format::OnDiskDuckDB => "TABLE",
            Format::Lance => {
                anyhow::bail!(
                    "Lance format is not supported for DuckDB engine. \
                    Please use lance-bench instead."
                );
            }
            format => anyhow::bail!("Format {format} isn't supported for DuckDB"),
        };

        // DuckDB loads from parquet for OnDiskDuckDB format
        let load_format = match file_format {
            Format::Parquet | Format::OnDiskDuckDB => Format::Parquet,
            f => f,
        };

        // Get the base URL for the format's data directory
        let format_url = benchmark.format_path(load_format, benchmark.data_url())?;
        let base_dir = format_url.as_str();
        let base_dir = base_dir
            .strip_prefix("file://")
            .unwrap_or(base_dir)
            .trim_end_matches('/');

        let commands =
            generate_duckdb_registration_sql(benchmark, base_dir, load_format, object_type);

        for stmt in commands {
            self.execute_query(&stmt)?;
        }

        Ok(())
    }

    /// Execute a query and return a `DuckQueryResult` wrapper.
    pub fn execute_query_result(&self, query: &str) -> Result<(Option<Duration>, DuckQueryResult)> {
        trace!("execute duckdb query: {query}");
        let time_instant = Instant::now();
        let result = self.connection().query(query)?;
        let query_time = time_instant.elapsed();
        Ok((Some(query_time), DuckQueryResult::from_query_result(result)))
    }
}

/// Eagerly materialized wrapper around DuckDB query results.
///
/// Materializes the result on construction so that both `row_count()`,
/// `display()`, and `result_rows()` can be called via shared reference.
pub struct DuckQueryResult {
    row_count: usize,
    display_string: String,
    column_names: Vec<String>,
    normalized_rows: Vec<Vec<String>>,
}

impl DuckQueryResult {
    /// Consume a DuckDB `QueryResult` and materialize its contents.
    pub fn from_query_result(result: QueryResult) -> Self {
        let row_count = usize::try_from(result.row_count()).unwrap_or(0);
        let col_count = usize::try_from(result.column_count()).unwrap_or(0);

        let mut column_names = Vec::with_capacity(col_count);
        for col_idx in 0..col_count {
            column_names.push(
                result
                    .column_name(col_idx)
                    .vortex_expect("column name should be valid")
                    .to_string(),
            );
        }

        let mut display_string = String::new();
        let mut normalized_rows = Vec::new();

        for chunk in result {
            let chunk_str =
                String::try_from(chunk.deref()).unwrap_or_else(|_| "<error>".to_string());
            display_string.push_str(&chunk_str);

            for row_idx in 0..chunk.len() {
                let mut row = Vec::with_capacity(chunk.column_count());
                for col_idx in 0..chunk.column_count() {
                    let vector = chunk.get_vector(col_idx);
                    let cell = match vector.get_value(row_idx, chunk.len()) {
                        Some(value) => value.to_string(),
                        None => "NULL".to_string(),
                    };
                    row.push(cell);
                }
                normalized_rows.push(row);
            }
        }

        Self {
            row_count,
            display_string,
            column_names,
            normalized_rows,
        }
    }
}

impl BenchmarkQueryResult for DuckQueryResult {
    fn row_count(&self) -> usize {
        self.row_count
    }

    fn display(self) -> String {
        self.display_string
    }

    fn result_rows(&self) -> (Vec<String>, Vec<Vec<String>>) {
        (self.column_names.clone(), self.normalized_rows.clone())
    }
}
