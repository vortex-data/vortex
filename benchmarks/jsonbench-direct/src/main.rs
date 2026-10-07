// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! JSONBench without a query engine.
//!
//! Runs the JSONBench queries as hand-written plans: each format extracts the queried JSON paths
//! in its own way, and all formats share one aggregation. The comparison isolates how fast each
//! format filters and extracts semi-structured data from what an engine adds on top.

mod agg;
mod parquet_scan;
mod vortex_scan;

use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;

use clap::Parser;
use clap::value_parser;
use vortex_bench::Benchmark;
use vortex_bench::BenchmarkArg;
use vortex_bench::Engine;
use vortex_bench::Format;
use vortex_bench::Opt;
use vortex_bench::Opts;
use vortex_bench::create_benchmark;
use vortex_bench::create_output_writer;
use vortex_bench::display::DisplayFormat;
use vortex_bench::require_prepared_data;
use vortex_bench::runner::BenchmarkMode;
use vortex_bench::runner::BenchmarkQueryResult;
use vortex_bench::runner::SqlBenchmarkRunner;
use vortex_bench::runner::filter_queries;
use vortex_bench::setup_logging_and_tracing;

use crate::agg::Query;
use crate::parquet_scan::Storage;

#[derive(Parser)]
#[command(about = "Run JSONBench queries over Vortex and Parquet without a query engine")]
struct Args {
    #[arg(short, long, default_value_t = 5)]
    iterations: usize,

    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Format), default_values = ["parquet", "parquet-variant", "vortex"])]
    formats: Vec<Format>,

    #[arg(short, long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,

    #[arg(short, long, value_delimiter = ',')]
    exclude_queries: Option<Vec<usize>>,

    #[arg(long = "opt", value_delimiter = ',', value_parser = value_parser!(Opt))]
    options: Vec<Opt>,

    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,

    #[arg(short, long)]
    output_path: Option<PathBuf>,

    #[arg(long, default_value = "local")]
    runner: String,

    #[arg(long)]
    hide_progress_bar: bool,

    #[arg(short, long)]
    verbose: bool,
}

/// Result rows of one query, already ordered and limited.
struct DirectResult(Vec<String>);

impl BenchmarkQueryResult for DirectResult {
    fn row_count(&self) -> usize {
        self.0.len()
    }

    fn display(self) -> String {
        agg::render(&self.0)
    }
}

/// The data files of `format`, sorted by name.
fn data_files(benchmark: &dyn Benchmark, format: Format) -> anyhow::Result<Vec<PathBuf>> {
    let dir = benchmark
        .format_path(format, benchmark.data_url())?
        .to_file_path()
        .map_err(|_| anyhow::anyhow!("jsonbench-direct reads local data only"))?;
    let pattern = dir.join(format!("*.{}", format.ext()));
    let mut files = glob::glob(&pattern.to_string_lossy())?.collect::<Result<Vec<_>, _>>()?;
    files.sort();
    anyhow::ensure!(!files.is_empty(), "no {format} files in {}", dir.display());
    Ok(files)
}

async fn run_query(
    query: Query,
    format: Format,
    files: Vec<PathBuf>,
) -> anyhow::Result<Vec<String>> {
    let partial = match format {
        Format::OnDiskVortex | Format::VortexCompact => vortex_scan::run(query, files).await?,
        Format::ParquetVariant => parquet_scan::run(query, Storage::Variant, files).await?,
        Format::Parquet => parquet_scan::run(query, Storage::Json, files).await?,
        format => anyhow::bail!("jsonbench-direct does not support {format}"),
    };
    Ok(partial.finish())
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    setup_logging_and_tracing(args.verbose, false)?;

    let benchmark = create_benchmark(BenchmarkArg::JsonBench, &Opts::from(args.options))?;
    let queries = filter_queries(
        benchmark.queries()?,
        args.queries.as_ref(),
        args.exclude_queries.as_ref(),
    );
    require_prepared_data(&*benchmark, &args.formats)?;

    let mut runner = SqlBenchmarkRunner::new(
        &*benchmark,
        Engine::Vortex,
        args.runner,
        args.formats.clone(),
        false,
        args.hide_progress_bar,
    )?;

    let benchmark_ref = &*benchmark;
    runner
        .run_all_async(
            &queries,
            BenchmarkMode::Run {
                iterations: args.iterations,
            },
            |format| async move { Ok((format, data_files(benchmark_ref, format)?)) },
            |query_idx, (format, files), _sql| {
                let format = *format;
                let files = files.clone();
                Box::pin(async move {
                    let query = Query::from_index(query_idx)?;
                    let start = Instant::now();
                    let rows = run_query(query, format, files).await?;
                    let elapsed: Duration = start.elapsed();
                    Ok((Some(elapsed), DirectResult(rows)))
                })
            },
        )
        .await?;

    let writer = create_output_writer(&args.display_format, args.output_path, "jsonbench-direct")?;
    runner.export_to(&args.display_format, writer)?;
    Ok(())
}
