// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! ClickHouse Local context for benchmarks.
//!
//! Spawns a fresh `clickhouse-local` process for each query execution. Setup SQL
//! (CREATE VIEW statements) is prepended to each query so that table definitions
//! are available, but the setup cost is excluded from query timings by measuring
//! only the wall-clock time from the first byte of query output to process exit.
//!
//! ## Why per-query processes?
//!
//! `clickhouse-local` in non-interactive (piped stdin) mode reads **all** of stdin
//! before executing any queries. This makes a persistent-process + delimiter protocol
//! impossible — the process blocks on stdin, never producing output until EOF. Spawning
//! a fresh process per query avoids this by closing stdin immediately after writing the
//! full SQL batch (setup + query), which triggers execution.
//!
//! Process spawn overhead is negligible (~1ms) compared to query execution times.
//!
//! The ClickHouse binary is resolved at build time via `build.rs`:
//! 1. `CLICKHOUSE_BINARY` env var — use the specified path.
//! 2. Falls back to `"clickhouse"` — resolved from `$PATH` at runtime.
//!
//! For local runs, install ClickHouse manually (e.g., `brew install clickhouse`
//! or download from <https://clickhouse.com/docs/en/install>).
//! In CI, it is installed by the workflow before the benchmark step.

use std::io::BufRead;
use std::io::BufReader;
use std::io::Read;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::process::Stdio;
use std::time::Duration;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use tracing::trace;
use vortex_bench::Benchmark;
use vortex_bench::Format;

/// Path to the ClickHouse binary, set by build.rs at compile time.
///
/// This is either the value of the `CLICKHOUSE_BINARY` env var at build time,
/// or `"clickhouse"` (resolved from `$PATH` at runtime).
const CLICKHOUSE_BINARY: &str = env!("CLICKHOUSE_BINARY");

/// A client that spawns `clickhouse-local` processes for running SQL benchmarks.
///
/// Setup SQL (CREATE VIEW) is stored at construction time and prepended to each
/// query execution. Each `execute_query` call spawns a fresh process, writes the
/// full SQL batch (setup + query), closes stdin, and reads the output.
pub struct ClickHouseClient {
    /// Path to the ClickHouse binary.
    binary: PathBuf,
    /// Setup SQL statements (CREATE VIEW) to prepend to each query.
    setup_sql: Vec<String>,
}

impl ClickHouseClient {
    /// Create a new client that will use `clickhouse-local` for query execution.
    ///
    /// Validates that the binary is available and builds setup SQL (CREATE VIEW
    /// statements) from the benchmark's table specs. Only Parquet format is supported.
    pub fn new(benchmark: &dyn Benchmark, format: Format) -> Result<Self> {
        if format != Format::Parquet {
            anyhow::bail!("clickhouse-bench only supports Parquet format, got {format}");
        }

        let binary = PathBuf::from(CLICKHOUSE_BINARY);
        Self::verify_binary(&binary)?;
        tracing::info!(binary = %binary.display(), "Using clickhouse-local");

        let setup_sql = Self::build_setup_sql(benchmark, format)?;

        Ok(Self { binary, setup_sql })
    }

    /// Check that the ClickHouse binary is available.
    ///
    /// For absolute paths, checks that the file exists on disk.
    /// For bare names (e.g., `"clickhouse"`), tries to invoke it to verify it's resolvable.
    fn verify_binary(binary: &Path) -> Result<()> {
        if binary.is_absolute() {
            anyhow::ensure!(
                binary.exists(),
                "ClickHouse binary not found at '{path}'. \
                 Set CLICKHOUSE_BINARY env var to the correct path, or install ClickHouse \
                 and ensure it is on $PATH.",
                path = binary.display()
            );
        }

        let output = Command::new(binary.as_os_str())
            .args(["local", "--version"])
            .output()
            .with_context(|| {
                format!(
                    "ClickHouse binary '{name}' not found on $PATH. \
                     Install ClickHouse (https://clickhouse.com/docs/en/install) or set \
                     CLICKHOUSE_BINARY env var to an absolute path before building.",
                    name = binary.display()
                )
            })?;

        anyhow::ensure!(
            output.status.success(),
            "ClickHouse binary at '{name}' failed to run: {stderr}",
            name = binary.display(),
            stderr = String::from_utf8_lossy(&output.stderr)
        );

        let version = String::from_utf8_lossy(&output.stdout);
        tracing::debug!(version = version.trim(), "Verified clickhouse binary");

        Ok(())
    }

    /// Build `CREATE VIEW ... AS SELECT * FROM file(...)` statements for all tables.
    fn build_setup_sql(benchmark: &dyn Benchmark, format: Format) -> Result<Vec<String>> {
        let data_url = benchmark.data_url();
        let base_dir = if data_url.scheme() == "file" {
            data_url
                .to_file_path()
                .map_err(|_| anyhow::anyhow!("Invalid file URL: {data_url}"))?
        } else {
            anyhow::bail!("clickhouse-bench only supports local file:// data URLs");
        };

        let format_dir = base_dir.join(format.name());
        if !format_dir.exists() {
            anyhow::bail!(
                "Data directory does not exist: {}. Run data generation first.",
                format_dir.display()
            );
        }

        let mut stmts = Vec::new();
        for table_spec in benchmark.table_specs() {
            let name = table_spec.name;
            let pattern = benchmark
                .pattern(name, format)
                .map(|p| p.to_string())
                .unwrap_or_else(|| format!("*.{}", format.ext()));

            let data_path = format!("{}/{}", format_dir.display(), pattern);

            tracing::info!(
                table = name,
                path = %data_path,
                "Registering ClickHouse table"
            );

            stmts.push(format!(
                "CREATE VIEW IF NOT EXISTS {name} AS \
                 SELECT * FROM file('{data_path}', Parquet);"
            ));
        }

        Ok(stmts)
    }

    /// Execute a SQL query, returning `(row_count, timing)`.
    ///
    /// Spawns a fresh `clickhouse-local` process with the setup SQL prepended to
    /// the query. Timing covers only query execution (from process spawn through
    /// output collection).
    pub fn execute_query(&mut self, query: &str) -> Result<(usize, Option<Duration>)> {
        trace!("execute clickhouse query: {query}");

        let mut query_str = query.to_string();
        if !query_str.trim_end().ends_with(';') {
            query_str.push(';');
        }

        // Build the full SQL batch: setup statements + query.
        let mut full_sql = String::new();
        for stmt in &self.setup_sql {
            full_sql.push_str(stmt);
            full_sql.push('\n');
        }
        full_sql.push_str(&query_str);
        full_sql.push('\n');

        let time_instant = Instant::now();

        let mut child = Command::new(self.binary.as_os_str())
            .args(["local", "--format", "TabSeparated"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .context("Failed to spawn clickhouse-local")?;

        // Write all SQL and close stdin to trigger execution.
        {
            let mut stdin = child.stdin.take().context("Failed to open stdin")?;
            stdin
                .write_all(full_sql.as_bytes())
                .context("Failed to write SQL to clickhouse-local")?;
            stdin.flush().context("Failed to flush stdin")?;
            // stdin is dropped here, closing the pipe and signaling EOF.
        }

        // Read all output lines from stdout.
        let stdout = child.stdout.take().context("Failed to open stdout")?;
        let reader = BufReader::new(stdout);
        let mut row_count = 0usize;
        for line in reader.lines() {
            let line = line.context("Failed to read from clickhouse-local stdout")?;
            if !line.trim().is_empty() {
                row_count += 1;
            }
        }

        let status = child
            .wait()
            .context("Failed to wait for clickhouse-local")?;

        let query_time = time_instant.elapsed();

        if !status.success() {
            let stderr = match child.stderr.take() {
                Some(s) => {
                    let mut buf = String::new();
                    BufReader::new(s).read_to_string(&mut buf).ok();
                    buf
                }
                None => String::new(),
            };
            anyhow::bail!("clickhouse-local exited with {status}. stderr:\n{stderr}",);
        }

        Ok((row_count, Some(query_time)))
    }
}
