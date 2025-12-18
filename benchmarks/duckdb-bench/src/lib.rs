// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DuckDB context for benchmarks.

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
use vortex_duckdb::duckdb::Config;
use vortex_duckdb::duckdb::Connection;
use vortex_duckdb::duckdb::Database;
use vortex_duckdb::register_extension_options;

/// DuckDB context for benchmarks.
pub struct DuckClient {
    pub db: Database,
    pub connection: Connection,
    pub db_path: PathBuf,
    pub threads: Option<usize>,
}

impl DuckClient {
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
            db,
            connection,
            db_path,
            threads,
        })
    }

    pub fn open_and_setup_database(
        path: Option<PathBuf>,
        threads: Option<usize>,
    ) -> Result<(Database, Connection)> {
        let config = Config::new().vortex_expect("failed to create duckdb config");

        // Register Vortex extension options before creating connection
        register_extension_options(&config);

        let db = match path {
            Some(path) => Database::open_with_config(path, config),
            None => Database::open_in_memory_with_config(config),
        }?;

        let connection = db.connect()?;
        vortex_duckdb::register_table_functions(&connection)?;

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

        // Set vortex_max_threads if specified
        if let Some(thread_count) = threads {
            connection.query(&format!("SET vortex_max_threads = {}", thread_count))?;
        }

        Ok((db, connection))
    }

    pub fn reopen(&mut self) -> Result<()> {
        // take ownership of the connection & database
        let mut connection = unsafe { Connection::borrow(self.connection.as_ptr()) };
        std::mem::swap(&mut self.connection, &mut connection);
        let mut db = unsafe { Database::borrow(self.db.as_ptr()) };
        std::mem::swap(&mut self.db, &mut db);

        // drop the connection, then the database (order might be important?)
        // NB: self.db and self.connection will be dangling pointers, which we'll fix below
        drop(connection);
        drop(db);

        let (mut db, mut connection) =
            Self::open_and_setup_database(Some(self.db_path.clone()), self.threads)?;

        std::mem::swap(&mut self.connection, &mut connection);
        std::mem::swap(&mut self.db, &mut db);

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
            db,
            connection,
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
        let result = self.connection.query(query)?;
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
}
