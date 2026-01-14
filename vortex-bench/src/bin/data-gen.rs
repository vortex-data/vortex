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
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::arrow::arrow_writer::ArrowWriter;
use tracing::info;
use vortex::error::VortexExpect;
use vortex_bench::Benchmark;
use vortex_bench::BenchmarkArg;
use vortex_bench::CompactionStrategy;
use vortex_bench::Format;
use vortex_bench::Opt;
use vortex_bench::Opts;
use vortex_bench::conversions::convert_parquet_to_vortex;
use vortex_bench::create_benchmark;
use vortex_bench::generate_duckdb_registration_sql;
use vortex_bench::setup_logging_and_tracing;

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

    // Generate base Parquet data - this is the source for all other formats
    benchmark.generate_base_data().await?;

    // Convert to other formats as needed (only for local file URLs)
    if benchmark.data_url().scheme() == "file" {
        let base_path = benchmark
            .data_url()
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", benchmark.data_url()))?;

        let repeat = opts.get_as::<usize>("repeat").unwrap_or(1);
        if repeat > 1 {
            repeat_parquet_files(&base_path, repeat).await?;
        }

        let repeat_single = opts.get_as::<usize>("repeat-single").unwrap_or(1);
        let only_filename =
            (repeat_single > 1).then(|| repeat_single_parquet(&base_path, repeat_single));
        let only_filename = match only_filename {
            Some(fut) => Some(fut.await?),
            None => None,
        };

        let strategy_builder = strategy_builder_from_opts(&opts);

        if args
            .formats
            .iter()
            .any(|f| matches!(f, Format::OnDiskVortex))
        {
            convert_parquet_to_vortex(
                &base_path,
                CompactionStrategy::Default,
                strategy_builder.clone(),
                only_filename.clone(),
            )
            .await?;
        }

        if args
            .formats
            .iter()
            .any(|f| matches!(f, Format::VortexCompact))
        {
            convert_parquet_to_vortex(
                &base_path,
                CompactionStrategy::Compact,
                strategy_builder.clone(),
                only_filename.clone(),
            )
            .await?;
        }

        if args
            .formats
            .iter()
            .any(|f| matches!(f, Format::OnDiskDuckDB))
        {
            generate_duckdb(&base_path, &*benchmark)?;
        }
    }

    Ok(())
}

fn strategy_builder_from_opts(opts: &Opts) -> Option<vortex::file::WriteStrategyBuilder> {
    let row_block_size = opts.get_as::<usize>("row-block-size");
    let block_size_min_mb = opts.get_as::<u64>("block-size-min-mb");
    let buffered_chunk_mb = opts.get_as::<u64>("buffered-chunk-mb");

    if row_block_size.is_none() && block_size_min_mb.is_none() && buffered_chunk_mb.is_none() {
        return None;
    }

    let mut builder = vortex::file::WriteStrategyBuilder::new();
    if let Some(row_block_size) = row_block_size {
        builder = builder.with_row_block_size(row_block_size);
    }
    if let Some(block_size_min_mb) = block_size_min_mb {
        builder = builder.with_block_size_minimum_bytes(block_size_min_mb << 20);
    }
    if let Some(buffered_chunk_mb) = buffered_chunk_mb {
        builder = builder.with_buffered_chunk_bytes(buffered_chunk_mb << 20);
    }

    Some(builder)
}

async fn repeat_parquet_files(base_path: &Path, repeat: usize) -> anyhow::Result<()> {
    let parquet_path = base_path.join(Format::Parquet.name());
    let entries = std::fs::read_dir(&parquet_path)?.collect::<std::io::Result<Vec<_>>>()?;
    let inputs = entries
        .iter()
        .filter(|entry| entry.path().extension().is_some_and(|e| e == "parquet"))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();

    if inputs.is_empty() {
        return Ok(());
    }

    for r in 1..repeat {
        for input in &inputs {
            let stem = input
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("data");
            let output = parquet_path.join(format!("{stem}_r{r}.parquet"));
            if output.exists() {
                continue;
            }
            tokio::fs::copy(input, output).await?;
        }
    }

    Ok(())
}

async fn repeat_single_parquet(base_path: &Path, repeat: usize) -> anyhow::Result<String> {
    let parquet_path = base_path.join(Format::Parquet.name());
    let entries = std::fs::read_dir(&parquet_path)?.collect::<std::io::Result<Vec<_>>>()?;
    let inputs = entries
        .iter()
        .filter(|entry| entry.path().extension().is_some_and(|e| e == "parquet"))
        .map(|entry| entry.path())
        .collect::<Vec<_>>();

    if inputs.len() != 1 {
        anyhow::bail!(
            "repeat-single expects exactly one parquet file, found {}",
            inputs.len()
        );
    }

    let input = inputs[0].clone();
    let stem = input
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("data");
    let output_name = format!("{stem}_repeat{repeat}.parquet");
    let output_path = parquet_path.join(&output_name);
    if output_path.exists() {
        return Ok(output_name);
    }

    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        let schema = ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(&input)?)?
            .schema()
            .clone();
        let mut writer = ArrowWriter::try_new(
            std::fs::File::create(&output_path)?,
            schema,
            None,
        )?;

        for _ in 0..repeat {
            let reader =
                ParquetRecordBatchReaderBuilder::try_new(std::fs::File::open(&input)?)?.build()?;
            for batch in reader {
                writer.write(&batch?)?;
            }
        }

        writer.close()?;
        Ok(())
    })
    .await??;

    Ok(output_name)
}

/// Generate a DuckDB database from Parquet files using the DuckDB CLI.
fn generate_duckdb(base_path: &Path, benchmark: &dyn Benchmark) -> anyhow::Result<()> {
    let duckdb_dir = base_path.join(Format::OnDiskDuckDB.name());
    std::fs::create_dir_all(&duckdb_dir)?;

    let db_path = duckdb_dir.join("duckdb.db");

    // Skip if database already exists
    if db_path.exists() {
        info!("DuckDB database already exists at {}", db_path.display());
        return Ok(());
    }

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
