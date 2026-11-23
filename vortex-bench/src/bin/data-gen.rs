// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark data generation binary.
//!
//! This binary generates benchmark data for all formats needed, consolidating
//! data generation that was previously duplicated across datafusion-bench and duckdb-bench.

use std::path::Path;
use std::process::Command;

use clap::Parser;
use clap::value_parser;
use mimalloc::MiMalloc;
use tracing::info;
use vortex::error::VortexExpect;
use vortex_bench::Benchmark;
use vortex_bench::BenchmarkArg;
use vortex_bench::CompactionStrategy;
use vortex_bench::Format;
use vortex_bench::LogFormat;
use vortex_bench::Opt;
use vortex_bench::Opts;
use vortex_bench::conversions::convert_parquet_directory_to_vortex;
use vortex_bench::create_benchmark;
use vortex_bench::generate_duckdb_registration_sql;
use vortex_bench::parquet_copies::SizeFormat;
use vortex_bench::parquet_copies::write_size_format;
use vortex_bench::setup_logging_and_tracing_with_format;

#[derive(Parser)]
#[command(name = "bench-data-gen")]
#[command(about = "Generate benchmark data for all requested formats")]
struct Args {
    #[arg(value_enum)]
    benchmark: BenchmarkArg,

    #[arg(short, long)]
    verbose: bool,

    #[arg(long)]
    tracing: bool,

    /// Format for the primary stderr log sink. `text` is the default human-readable format;
    /// `json` emits one JSON object per event, suitable for piping into `jq`.
    #[arg(long, value_enum, default_value_t = LogFormat::Text)]
    log_format: LogFormat,

    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Format))]
    formats: Vec<Format>,

    /// Formats to write only for their file sizes, from the base Parquet files. No query reads
    /// them.
    #[arg(long, value_delimiter = ',', value_enum)]
    size_formats: Vec<SizeFormat>,

    #[arg(long = "opt", value_delimiter = ',', value_parser = value_parser!(Opt))]
    options: Vec<Opt>,
}

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    vortex_bench::configure_allocator();
    let args = Args::parse();
    let opts = Opts::from(args.options);

    setup_logging_and_tracing_with_format(args.verbose, args.tracing, args.log_format)?;

    let benchmark = create_benchmark(args.benchmark, &opts)?;

    // Generate base Parquet data - this is the source for all other formats
    benchmark.generate_base_data().await?;

    // Convert to other formats as needed (only for local file URLs)
    if benchmark.data_url().scheme() == "file" {
        let base_path = benchmark
            .data_url()
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", benchmark.data_url()))?;

        if args
            .formats
            .iter()
            .any(|f| matches!(f, Format::OnDiskVortex))
        {
            convert_parquet_directory_to_vortex(&base_path, CompactionStrategy::Default).await?;
        }

        if args
            .formats
            .iter()
            .any(|f| matches!(f, Format::VortexCompact))
        {
            convert_parquet_directory_to_vortex(&base_path, CompactionStrategy::Compact).await?;
        }

        if args
            .formats
            .iter()
            .any(|f| matches!(f, Format::OnDiskDuckDB))
        {
            generate_duckdb(&base_path, &*benchmark)?;
        }

        for format in args.formats.iter().copied() {
            benchmark.prepare_format(format, &base_path).await?;
        }

        for format in args.size_formats.iter().copied() {
            let base_path = base_path.clone();
            tokio::task::spawn_blocking(move || write_size_format(&base_path, format)).await??;
        }
    }

    Ok(())
}

/// Generate a DuckDB database from Parquet files using the DuckDB CLI.
fn generate_duckdb(base_path: &Path, benchmark: &dyn Benchmark) -> anyhow::Result<()> {
    let duckdb_dir = base_path.join(Format::OnDiskDuckDB.name());
    let parquet_dir = base_path.join(Format::Parquet.name());
    let sql = generate_duckdb_registration_sql(
        benchmark,
        parquet_dir
            .to_str()
            .vortex_expect("value must be str displayable"),
        Format::Parquet,
        "TABLE",
    );

    run_duckdb_statements(&duckdb_dir, &sql, |path, stmt| {
        let output = Command::new("duckdb")
            .arg(path)
            .arg("-c")
            .arg(stmt)
            .output()?;
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("DuckDB CLI failed: {}", stderr);
        }
        Ok(())
    })
}

fn run_duckdb_statements(
    duckdb_dir: &Path,
    sql: &[String],
    mut run_stmt: impl FnMut(&Path, &str) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    std::fs::create_dir_all(duckdb_dir)?;

    let db_path = duckdb_dir.join("duckdb.db");
    let tmp_path = duckdb_dir.join("duckdb.db.tmp");
    let tmp_wal_path = duckdb_dir.join("duckdb.db.tmp.wal");

    // Remove any stale staging files from an earlier interrupted run.
    for path in [&tmp_path, &tmp_wal_path] {
        if let Err(err) = std::fs::remove_file(path)
            && err.kind() != std::io::ErrorKind::NotFound
        {
            return Err(err.into());
        }
    }

    // Re-run `CREATE TABLE IF NOT EXISTS` against an existing database so a
    // pre-existing partial database gets its missing tables without rebuilding.
    if db_path.exists() {
        info!(
            "DuckDB database already exists at {}; ensuring all tables are imported",
            db_path.display()
        );
        for stmt in sql {
            run_stmt(&db_path, stmt)?;
        }
        return Ok(());
    }

    for stmt in sql {
        if let Err(err) = run_stmt(&tmp_path, stmt) {
            drop(std::fs::remove_file(&tmp_path));
            drop(std::fs::remove_file(&tmp_wal_path));
            return Err(err);
        }
    }

    std::fs::rename(&tmp_path, &db_path)?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::io::Write;

    use super::*;

    fn two_table_sql() -> Vec<String> {
        vec![
            "CREATE TABLE IF NOT EXISTS first_tbl AS SELECT 1;".to_string(),
            "CREATE TABLE IF NOT EXISTS second_tbl AS SELECT 2;".to_string(),
        ]
    }

    fn append_table_if_missing(path: &Path, stmt: &str) -> anyhow::Result<()> {
        let existing = if path.exists() {
            std::fs::read_to_string(path)?
        } else {
            String::new()
        };
        if !existing.lines().any(|line| line == stmt) {
            let mut file = std::fs::OpenOptions::new()
                .create(true)
                .append(true)
                .open(path)?;
            writeln!(file, "{stmt}")?;
        }
        Ok(())
    }

    #[test]
    fn failed_second_table_stages_and_retries_cleanly() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let duckdb_dir = dir.path().join("duckdb");
        let sql = two_table_sql();

        let err = run_duckdb_statements(&duckdb_dir, &sql, |path, stmt| {
            if stmt.contains("second_tbl") {
                anyhow::bail!("missing second_tbl parquet");
            }
            append_table_if_missing(path, stmt)
        })
        .unwrap_err();
        assert!(err.to_string().contains("missing second_tbl"));
        assert!(!duckdb_dir.join("duckdb.db").exists());
        assert!(!duckdb_dir.join("duckdb.db.tmp").exists());

        run_duckdb_statements(&duckdb_dir, &sql, append_table_if_missing)?;
        let contents = std::fs::read_to_string(duckdb_dir.join("duckdb.db"))?;
        assert!(contents.contains("first_tbl"));
        assert!(contents.contains("second_tbl"));
        Ok(())
    }

    #[test]
    fn existing_partial_database_imports_missing_table() -> anyhow::Result<()> {
        let dir = tempfile::tempdir()?;
        let duckdb_dir = dir.path().join("duckdb");
        std::fs::create_dir_all(&duckdb_dir)?;
        let sql = two_table_sql();
        let db_path = duckdb_dir.join("duckdb.db");
        std::fs::write(&db_path, format!("{}\n", sql[0]))?;

        run_duckdb_statements(&duckdb_dir, &sql, append_table_if_missing)?;
        let contents = std::fs::read_to_string(&db_path)?;
        assert_eq!(contents.lines().count(), 2);
        assert!(contents.contains("first_tbl"));
        assert!(contents.contains("second_tbl"));
        Ok(())
    }
}
