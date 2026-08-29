// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! DuckDB benchmark client that loads Vortex as an out-of-tree extension.

use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::c_char;
use std::ffi::c_void;
use std::path::Path;
use std::path::PathBuf;
use std::ptr;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use tracing::trace;
use vortex_bench::Benchmark;
use vortex_bench::Format;
use vortex_bench::IdempotentPath;
use vortex_bench::generate_duckdb_registration_sql;
use vortex_bench::runner::BenchmarkQueryResult;

type DuckDatabase = *mut c_void;
type DuckConnection = *mut c_void;
type DuckConfig = *mut c_void;

#[repr(C)]
struct DuckResult {
    deprecated_column_count: u64,
    deprecated_row_count: u64,
    deprecated_rows_changed: u64,
    deprecated_columns: *mut c_void,
    deprecated_error_message: *mut c_char,
    internal_data: *mut c_void,
}

unsafe extern "C" {
    fn duckdb_create_config(config: *mut DuckConfig) -> u32;
    fn duckdb_set_config(config: DuckConfig, name: *const c_char, value: *const c_char) -> u32;
    fn duckdb_destroy_config(config: *mut DuckConfig);
    fn duckdb_open_ext(
        path: *const c_char,
        database: *mut DuckDatabase,
        config: DuckConfig,
        error: *mut *mut c_char,
    ) -> u32;
    fn duckdb_close(database: *mut DuckDatabase);
    fn duckdb_connect(database: DuckDatabase, connection: *mut DuckConnection) -> u32;
    fn duckdb_disconnect(connection: *mut DuckConnection);
    fn duckdb_query(
        connection: DuckConnection,
        query: *const c_char,
        result: *mut DuckResult,
    ) -> u32;
    fn duckdb_result_error(result: *mut DuckResult) -> *const c_char;
    fn duckdb_destroy_result(result: *mut DuckResult);
    fn duckdb_row_count(result: *mut DuckResult) -> u64;
    fn duckdb_rows_changed(result: *mut DuckResult) -> u64;
    fn duckdb_free(ptr: *mut c_void);
}

/// DuckDB context for benchmarks.
pub struct DuckClient {
    db: Option<DuckDatabase>,
    connection: Option<DuckConnection>,
    pub db_path: PathBuf,
    pub threads: Option<usize>,
    init_sql: Vec<String>,
}

impl DuckClient {
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
        let db_path = dir.join("duckdb.db");

        if format != Format::OnDiskDuckDB {
            std::fs::create_dir_all(&dir)?;
        } else if data_url.scheme() != "file" {
            anyhow::bail!("DuckDB format requires local data prepared by data-gen");
        } else if !db_path.exists() {
            anyhow::bail!(
                "prepared DuckDB database is missing at {}",
                db_path.display()
            );
        }
        if delete_database && db_path.exists() && format != Format::OnDiskDuckDB {
            std::fs::remove_file(&db_path)?;
        }

        let (db, connection) = Self::open_and_setup_database(Some(&db_path), threads)?;
        Ok(Self {
            db: Some(db),
            connection: Some(connection),
            db_path,
            threads,
            init_sql: Vec::new(),
        })
    }

    fn query(&self, query: &str) -> Result<DuckQueryResult> {
        let query = CString::new(query).context("query contains a NUL byte")?;
        let mut result: DuckResult = unsafe { std::mem::zeroed() };
        let status = unsafe {
            duckdb_query(
                self.connection.context("DuckDB connection is closed")?,
                query.as_ptr(),
                &raw mut result,
            )
        };
        if status != 0 {
            let error = result_error(&mut result);
            unsafe { duckdb_destroy_result(&raw mut result) };
            anyhow::bail!("failed to execute query: {error}");
        }
        Ok(DuckQueryResult(result))
    }

    fn open_and_setup_database(
        path: Option<&Path>,
        threads: Option<usize>,
    ) -> Result<(DuckDatabase, DuckConnection)> {
        let mut config = ptr::null_mut();
        if unsafe { duckdb_create_config(&raw mut config) } != 0 {
            anyhow::bail!("failed to create DuckDB config");
        }
        let option = CString::new("allow_unsigned_extensions")?;
        let enabled = CString::new("true")?;
        if unsafe { duckdb_set_config(config, option.as_ptr(), enabled.as_ptr()) } != 0 {
            unsafe { duckdb_destroy_config(&raw mut config) };
            anyhow::bail!("failed to enable unsigned DuckDB extensions");
        }

        let path = path
            .map(|path| CString::new(path.to_string_lossy().as_bytes()))
            .transpose()?;
        let mut db = ptr::null_mut();
        let mut error = ptr::null_mut();
        let status = unsafe {
            duckdb_open_ext(
                path.as_ref().map_or(ptr::null(), |path| path.as_ptr()),
                &raw mut db,
                config,
                &raw mut error,
            )
        };
        unsafe { duckdb_destroy_config(&raw mut config) };
        if status != 0 {
            let message = if error.is_null() {
                "unknown DuckDB open error".to_string()
            } else {
                let message = unsafe { CStr::from_ptr(error) }
                    .to_string_lossy()
                    .into_owned();
                unsafe { duckdb_free(error.cast()) };
                message
            };
            anyhow::bail!("failed to open DuckDB: {message}");
        }

        let mut connection = ptr::null_mut();
        if unsafe { duckdb_connect(db, &raw mut connection) } != 0 {
            unsafe { duckdb_close(&raw mut db) };
            anyhow::bail!("failed to connect to DuckDB");
        }
        let extension = std::env::var("VORTEX_DUCKDB_EXTENSION")
            .context("VORTEX_DUCKDB_EXTENSION must point at the loadable extension")?;
        query_raw(
            connection,
            &format!("LOAD '{}'", extension.replace('\'', "''")),
        )?;
        if let Some(thread_count) = threads {
            query_raw(connection, &format!("SET threads = {thread_count}"))?;
        }
        query_raw(connection, "SET parquet_metadata_cache = true")?;
        Ok((db, connection))
    }

    pub fn set_init_sql(&mut self, statements: Vec<String>) -> Result<()> {
        for statement in &statements {
            self.query(statement)?;
        }
        self.init_sql = statements;
        Ok(())
    }

    pub fn reopen(&mut self) -> Result<()> {
        self.close();
        let (db, connection) = Self::open_and_setup_database(Some(&self.db_path), self.threads)?;
        self.db = Some(db);
        self.connection = Some(connection);
        for statement in &self.init_sql {
            self.query(statement)?;
        }
        Ok(())
    }

    pub fn new_in_memory() -> Result<Self> {
        let dir = std::env::temp_dir().join("vortex-duckdb-bench/in-memory");
        std::fs::create_dir_all(&dir)?;
        let db_path = dir.join("duckdb.db");
        let (db, connection) = Self::open_and_setup_database(Some(&db_path), None)?;
        Ok(Self {
            db: Some(db),
            connection: Some(connection),
            db_path,
            threads: None,
            init_sql: Vec::new(),
        })
    }

    pub fn execute_query(&self, query: &str) -> Result<(usize, Option<Duration>)> {
        trace!("execute duckdb query: {query}");
        let start = Instant::now();
        let result = self.query(query)?;
        Ok((result.row_count(), Some(start.elapsed())))
    }

    pub fn register_tables<B: Benchmark + ?Sized>(
        &self,
        benchmark: &B,
        file_format: Format,
    ) -> Result<()> {
        if file_format == Format::OnDiskDuckDB {
            return Ok(());
        }
        let object_type = match file_format {
            Format::Parquet
            | Format::OnDiskVortex
            | Format::VortexCompact
            | Format::VortexSpatialNative => "VIEW",
            Format::OnDiskDuckDB => "TABLE",
            format => anyhow::bail!("Format {format} isn't supported for DuckDB"),
        };
        let format_url = benchmark.format_path(file_format, benchmark.data_url())?;
        let base_dir = format_url
            .as_str()
            .strip_prefix("file://")
            .unwrap_or(format_url.as_str())
            .trim_end_matches('/');
        for statement in
            generate_duckdb_registration_sql(benchmark, base_dir, file_format, object_type)
        {
            self.query(&statement)?;
        }
        Ok(())
    }

    pub fn execute_query_result(&self, query: &str) -> Result<(Option<Duration>, DuckQueryResult)> {
        trace!("execute duckdb query: {query}");
        let start = Instant::now();
        let result = self.query(query)?;
        Ok((Some(start.elapsed()), result))
    }

    fn close(&mut self) {
        if let Some(mut connection) = self.connection.take() {
            unsafe { duckdb_disconnect(&raw mut connection) };
        }
        if let Some(mut db) = self.db.take() {
            unsafe { duckdb_close(&raw mut db) };
        }
    }
}

impl Drop for DuckClient {
    fn drop(&mut self) {
        self.close();
    }
}

fn query_raw(connection: DuckConnection, query: &str) -> Result<()> {
    let query = CString::new(query)?;
    let mut result: DuckResult = unsafe { std::mem::zeroed() };
    let status = unsafe { duckdb_query(connection, query.as_ptr(), &raw mut result) };
    if status != 0 {
        let error = result_error(&mut result);
        unsafe { duckdb_destroy_result(&raw mut result) };
        anyhow::bail!("failed to execute query: {error}");
    }
    unsafe { duckdb_destroy_result(&raw mut result) };
    Ok(())
}

fn result_error(result: &mut DuckResult) -> String {
    unsafe {
        let error = duckdb_result_error(result);
        if error.is_null() {
            "unknown DuckDB error".to_string()
        } else {
            CStr::from_ptr(error).to_string_lossy().into_owned()
        }
    }
}

pub struct DuckQueryResult(DuckResult);

impl DuckQueryResult {
    fn row_count(&self) -> usize {
        let result = (&raw const self.0).cast_mut();
        let changed = unsafe { duckdb_rows_changed(result) };
        usize::try_from(if changed == 0 {
            unsafe { duckdb_row_count(result) }
        } else {
            changed
        })
        .unwrap_or(0)
    }
}

impl Drop for DuckQueryResult {
    fn drop(&mut self) {
        unsafe { duckdb_destroy_result(&raw mut self.0) };
    }
}

impl BenchmarkQueryResult for DuckQueryResult {
    fn row_count(&self) -> usize {
        self.row_count()
    }

    fn display(self) -> String {
        format!("{} rows", self.row_count())
    }
}
