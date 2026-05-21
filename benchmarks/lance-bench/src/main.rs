// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use clap::value_parser;
use lance::datafusion::LanceTableProvider;
use lance::dataset::Dataset;
use lance::deps::arrow_array::RecordBatch;
use lance::deps::datafusion::arrow::util::pretty::pretty_format_batches;
use lance::deps::datafusion::physical_plan::ExecutionPlan;
use lance::deps::datafusion::prelude::SessionContext;
use lance_bench::convert::convert_parquet_to_lance;
use tracing::info;
use vortex_bench::Benchmark;
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
use vortex_bench::v3;

/// Lance benchmark tool - runs SQL queries against Lance format data using DataFusion
#[derive(Parser)]
struct Args {
    #[arg(value_enum)]
    benchmark: BenchmarkArg,

    #[arg(short, long, default_value_t = 5)]
    iterations: usize,

    #[arg(short, long)]
    threads: Option<usize>,

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

    /// Additionally write v3 JSONL records to this path. See
    /// `benchmarks-website/planning/02-contracts.md`.
    #[arg(long)]
    gh_json_v3: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    hide_progress_bar: bool,

    #[arg(long, default_value_t = false)]
    track_memory: bool,

    #[arg(long, default_value = "unknown")]
    runner: String,

    #[arg(long = "opt", value_delimiter = ',', value_parser = value_parser!(Opt))]
    options: Vec<Opt>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let opts = Opts::from(args.options);

    setup_logging_and_tracing(args.verbose, args.tracing)?;

    let benchmark = create_benchmark(args.benchmark, &opts)?;

    let filtered_queries = filter_queries(
        benchmark.queries()?,
        args.queries.as_ref(),
        args.exclude_queries.as_ref(),
    );

    // Generate base Parquet data first
    benchmark.generate_base_data().await?;

    // Convert Parquet to Lance format
    generate_lance_data(&*benchmark).await?;

    let mut runner = SqlBenchmarkRunner::new(
        &*benchmark,
        Engine::DataFusion,
        args.runner.clone(),
        vec![Format::Lance],
        args.track_memory,
        args.hide_progress_bar,
    )?;

    runner
        .run_all_async(
            &filtered_queries,
            BenchmarkMode::Run {
                iterations: args.iterations,
            },
            |_format| async {
                let session = SessionContext::new();
                register_lance_tables(&session, &*benchmark).await?;
                Ok(session)
            },
            |_query_idx, session, query| {
                Box::pin(async move {
                    let timer = Instant::now();
                    let (batches, _plan) = execute_query(session, query).await?;
                    let time = timer.elapsed();
                    anyhow::Ok((Some(time), LanceQueryResult(batches)))
                })
            },
        )
        .await?;

    if let Some(path) = args.gh_json_v3.as_ref() {
        v3::write_jsonl_to_path(path, &runner.v3_records())?;
    }

    let benchmark_id = format!("lance-{}", benchmark.dataset_name());
    let writer = create_output_writer(&args.display_format, args.output_path, &benchmark_id)?;
    runner.export_to(&args.display_format, writer)?;

    Ok(())
}

async fn register_lance_tables<B: Benchmark + ?Sized>(
    session: &SessionContext,
    benchmark: &B,
) -> anyhow::Result<()> {
    let benchmark_base = benchmark
        .data_url()
        .join(&format!("{}/", Format::Lance.name()))?;

    for table in benchmark.table_specs().iter() {
        let table_path = benchmark_base.join(&format!("{}.lance/", table.name))?;

        let dataset = Dataset::open(table_path.as_str()).await?;
        let provider = LanceTableProvider::new(
            Arc::new(dataset),
            false, // with_row_id
            false, // with_row_addr
        );

        session.register_table(table.name, Arc::new(provider))?;
    }

    Ok(())
}

/// Wrapper around Lance/DataFusion record batches implementing `BenchmarkQueryResult`.
struct LanceQueryResult(Vec<RecordBatch>);

impl BenchmarkQueryResult for LanceQueryResult {
    fn row_count(&self) -> usize {
        self.0.iter().map(|batch| batch.num_rows()).sum()
    }

    fn display(self) -> String {
        // Lance uses the same Arrow RecordBatch type
        pretty_format_batches(&self.0)
            .map(|d| d.to_string())
            .unwrap_or_else(|e| format!("<error: {e}>"))
    }
}

pub async fn execute_query(
    ctx: &SessionContext,
    query: &str,
) -> anyhow::Result<(Vec<RecordBatch>, Arc<dyn ExecutionPlan>)> {
    let df = ctx.sql(query).await?;

    let physical_plan = df.clone().create_physical_plan().await?;
    let result = df.collect().await?;

    Ok((result, physical_plan))
}

/// Generate Lance data from Parquet base data for the given benchmark.
async fn generate_lance_data<B: Benchmark + ?Sized>(benchmark: &B) -> anyhow::Result<()> {
    let data_url = benchmark.data_url();

    // Skip if using remote storage
    if data_url.scheme() != "file" {
        info!("Using remote data URL, assuming Lance data already exists");
        return Ok(());
    }

    let base_path = data_url
        .to_file_path()
        .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", data_url))?;

    let parquet_dir = base_path.join(Format::Parquet.name());
    let lance_dir = base_path.join(Format::Lance.name());

    info!(
        "Converting Parquet data from {} to Lance format at {}",
        parquet_dir.display(),
        lance_dir.display()
    );

    // Convert each table to Lance format
    for table in benchmark.table_specs().iter() {
        // Determine file prefix pattern for this table
        // TPC-H/TPC-DS use {table}_ prefix, others may use the table name directly
        let file_prefix = benchmark
            .pattern(table.name, Format::Parquet)
            .and_then(|p| {
                // Extract prefix from pattern like "customer_*.parquet" -> "customer_"
                let pattern_str = p.as_str();
                pattern_str
                    .strip_suffix(&format!("*.{}", Format::Parquet.ext()))
                    .map(|s| s.to_string())
            });

        convert_parquet_to_lance(
            &parquet_dir,
            &lance_dir,
            table.name,
            file_prefix.as_deref(),
            true, // Convert Utf8View to Utf8 for Lance compatibility
        )
        .await?;
    }

    Ok(())
}
