// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark data generation binary.
//!
//! This binary generates benchmark data for all formats needed, consolidating
//! data generation that was previously duplicated across datafusion-bench and duckdb-bench.

use clap::Parser;
use clap::value_parser;
use mimalloc::MiMalloc;
use vortex_bench::BenchmarkArg;
use vortex_bench::CompactionStrategy;
use vortex_bench::Format;
use vortex_bench::LogFormat;
use vortex_bench::Opt;
use vortex_bench::Opts;
use vortex_bench::conversions::convert_parquet_directory_to_vortex;
use vortex_bench::create_benchmark;
use vortex_bench::generate_duckdb;
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
