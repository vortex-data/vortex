// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;

use clap::Parser;
use clickhouse_bench::ClickHouseClient;
use tokio::runtime::Runtime;
use vortex_bench::BenchmarkArg;
use vortex_bench::Engine;
use vortex_bench::Format;
use vortex_bench::Opt;
use vortex_bench::Opts;
use vortex_bench::create_benchmark;
use vortex_bench::create_output_writer;
use vortex_bench::display::DisplayFormat;
use vortex_bench::runner::BenchmarkMode;
use vortex_bench::runner::BenchmarkQueryResult;
use vortex_bench::runner::SqlBenchmarkRunner;
use vortex_bench::runner::filter_queries;
use vortex_bench::setup_logging_and_tracing;

/// ClickHouse (clickhouse-local) benchmark runner.
///
/// Runs queries against Parquet data using clickhouse-local as a performance baseline.
/// This allows comparing ClickHouse's native Parquet reading performance against other engines
/// (DuckDB, DataFusion) on the same hardware and dataset.
#[derive(Parser)]
struct Args {
    #[arg(value_enum)]
    benchmark: BenchmarkArg,

    #[arg(short, long, default_value_t = 5)]
    iterations: usize,

    #[arg(short, long)]
    verbose: bool,

    #[arg(long)]
    tracing: bool,

    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,

    #[arg(short, long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,

    #[arg(short, long, value_delimiter = ',')]
    exclude_queries: Option<Vec<usize>>,

    #[arg(short)]
    output_path: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    track_memory: bool,

    #[arg(long, default_value_t = false)]
    hide_progress_bar: bool,

    #[arg(long = "opt", value_delimiter = ',', value_parser = clap::value_parser!(Opt))]
    options: Vec<Opt>,
}

struct ClickHouseQueryResult {
    row_count: usize,
}

impl BenchmarkQueryResult for ClickHouseQueryResult {
    fn row_count(&self) -> usize {
        self.row_count
    }

    fn display(self) -> String {
        format!("{} rows", self.row_count)
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let opts = Opts::from(args.options);

    setup_logging_and_tracing(args.verbose, args.tracing)?;

    let benchmark = create_benchmark(args.benchmark, &opts)?;

    let filtered_queries = filter_queries(
        benchmark.queries()?,
        args.queries.as_ref(),
        args.exclude_queries.as_ref(),
    );

    // Generate base Parquet data if needed.
    if benchmark.data_url().scheme() == "file" {
        let runtime = Runtime::new()?;
        runtime.block_on(async { benchmark.generate_base_data().await })?;
    }

    let formats = vec![Format::Parquet];

    let mut runner = SqlBenchmarkRunner::new(
        benchmark.as_ref(),
        Engine::ClickHouse,
        formats,
        args.track_memory,
        args.hide_progress_bar,
    )?;

    runner.run_all(
        &filtered_queries,
        BenchmarkMode::Run {
            iterations: args.iterations,
        },
        |format| ClickHouseClient::new(benchmark.as_ref(), format),
        |ctx, _query_idx, _format, query| {
            let (row_count, duration) = ctx.execute_query(query)?;
            Ok((duration, ClickHouseQueryResult { row_count }))
        },
    )?;

    let benchmark_id = format!("clickhouse-{}", benchmark.dataset_name());
    let writer = create_output_writer(&args.display_format, args.output_path, &benchmark_id)?;
    runner.export_to(&args.display_format, writer)?;

    Ok(())
}
