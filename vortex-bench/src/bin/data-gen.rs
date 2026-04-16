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
use vortex::error::VortexExpect;
use vortex_bench::Benchmark;
use vortex_bench::BenchmarkArg;
use vortex_bench::CompactionStrategy;
use vortex_bench::Format;
use vortex_bench::Opt;
use vortex_bench::Opts;
use vortex_bench::conversions::convert_parquet_directory_to_vortex;
use vortex_bench::create_benchmark;
use vortex_bench::generate_duckdb_registration_sql;
use vortex_bench::setup_logging_and_tracing;
use vortex_bench::utils::file::idempotent_dir;
use vortex_bench::utils::file::idempotent_dir_async;

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

    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Format))]
    formats: Vec<Format>,

    #[arg(long = "opt", value_delimiter = ',', value_parser = value_parser!(Opt))]
    options: Vec<Opt>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let opts = Opts::from(args.options);

    setup_logging_and_tracing(args.verbose, args.tracing)?;

    let benchmark = create_benchmark(args.benchmark, &opts)?;

    // All conversions are only meaningful for local file URLs.
    if benchmark.data_url().scheme() != "file" {
        return Ok(());
    }

    let base_path = benchmark
        .data_url()
        .to_file_path()
        .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", benchmark.data_url()))?;

    let parquet_dir = base_path.join(Format::Parquet.name());
    idempotent_dir_async(&parquet_dir, || benchmark.generate_base_data()).await?;

    if args.formats.iter().any(|f| matches!(f, Format::OnDiskVortex)) {
        let vortex_dir = base_path.join(Format::OnDiskVortex.name());
        idempotent_dir_async(&vortex_dir, || {
            convert_parquet_directory_to_vortex(&base_path, CompactionStrategy::Default)
        })
        .await?;
    }

    if args.formats.iter().any(|f| matches!(f, Format::VortexCompact)) {
        let compact_dir = base_path.join(Format::VortexCompact.name());
        idempotent_dir_async(&compact_dir, || {
            convert_parquet_directory_to_vortex(&base_path, CompactionStrategy::Compact)
        })
        .await?;
    }

    if args.formats.iter().any(|f| matches!(f, Format::OnDiskDuckDB)) {
        let duckdb_dir = base_path.join(Format::OnDiskDuckDB.name());
        idempotent_dir(&duckdb_dir, || generate_duckdb(&base_path, &*benchmark))?;
    }

    Ok(())
}

/// Generate a DuckDB database from Parquet files using the DuckDB CLI.
fn generate_duckdb(base_path: &Path, benchmark: &dyn Benchmark) -> anyhow::Result<()> {
    let duckdb_dir = base_path.join(Format::OnDiskDuckDB.name());
    std::fs::create_dir_all(&duckdb_dir)?;

    let db_path = duckdb_dir.join("duckdb.db");
    let parquet_dir = base_path.join(Format::Parquet.name());
    let sql = generate_duckdb_registration_sql(
        benchmark,
        parquet_dir
            .to_str()
            .vortex_expect("value must be str displayable"),
        Format::Parquet,
        "TABLE",
    );

    for stmt in sql.into_iter() {
        let output = Command::new("duckdb")
            .arg(&db_path)
            .arg("-c")
            .arg(&stmt)
            .output()?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            anyhow::bail!("DuckDB CLI failed: {}", stderr);
        }
    }

    Ok(())
}
