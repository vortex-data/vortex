// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use clap::value_parser;
use datafusion::arrow::array::RecordBatch;
use datafusion::datasource::listing::ListingOptions;
use datafusion::datasource::listing::ListingTable;
use datafusion::datasource::listing::ListingTableConfig;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::parquet::arrow::ParquetRecordBatchStreamBuilder;
use datafusion::prelude::SessionContext;
use datafusion_bench::format_to_df_format;
use datafusion_bench::metrics::MetricsSetExt;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::collect;
use futures::StreamExt;
use parking_lot::Mutex;
use tokio::fs::File;
use vortex_bench::Benchmark;
use vortex_bench::BenchmarkArg;
use vortex_bench::CompactionStrategy;
use vortex_bench::Engine;
use vortex_bench::Format;
use vortex_bench::Opt;
use vortex_bench::Opts;
use vortex_bench::conversions::convert_parquet_to_vortex;
use vortex_bench::create_benchmark;
use vortex_bench::create_output_writer;
use vortex_bench::display::DisplayFormat;
use vortex_bench::runner::SqlBenchmarkRunner;
use vortex_bench::runner::filter_queries;
use vortex_bench::setup_logging_and_tracing;
use vortex_datafusion::metrics::VortexMetricsFinder;

/// Common arguments shared across benchmarks
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

    #[arg(long)]
    export_spans: bool,

    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,

    #[arg(long, default_value_t = false)]
    delete_duckdb_database: bool,

    #[arg(short, long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,

    #[arg(short, long, value_delimiter = ',')]
    exclude_queries: Option<Vec<usize>>,

    #[arg(short)]
    output_path: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    show_metrics: bool,

    #[arg(long, default_value_t = false)]
    hide_progress_bar: bool,

    #[arg(long, default_value_t = false)]
    emit_plan: bool,

    #[arg(long, default_value_t = false)]
    track_memory: bool,

    #[arg(long, default_value_t = false)]
    explain: bool,

    #[arg(long, default_value_t = false)]
    explain_analyze: bool,

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

    let filtered_queries = filter_queries(
        benchmark.queries()?,
        args.queries.as_ref(),
        args.exclude_queries.as_ref(),
    );

    // Generate Vortex files from Parquet for any Vortex formats requested
    if benchmark.data_url().scheme() == "file" {
        benchmark.generate_base_data().await?;

        let base_path = benchmark
            .data_url()
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("Invalid file URL: {}", benchmark.data_url()))?;

        for format in args.formats.iter() {
            match format {
                Format::OnDiskVortex => {
                    convert_parquet_to_vortex(&base_path, CompactionStrategy::Default).await?;
                }
                Format::VortexCompact => {
                    convert_parquet_to_vortex(&base_path, CompactionStrategy::Compact).await?;
                }
                _ => {}
            }
        }
    }

    let mut runner = SqlBenchmarkRunner::new(
        &*benchmark,
        Engine::DataFusion,
        args.formats.clone(),
        args.track_memory,
        args.hide_progress_bar,
    )?;

    // Collect execution plans for metrics if show_metrics is enabled
    // Structure: (query_idx, format, execution_plan)
    #[allow(clippy::type_complexity)]
    let collected_plans: Arc<Mutex<Vec<(usize, Format, Arc<dyn ExecutionPlan>)>>> =
        Arc::new(Mutex::new(Vec::new()));
    let show_metrics = args.show_metrics;

    runner
        .run_all_async(
            &filtered_queries,
            args.iterations,
            |format| {
                let benchmark = &*benchmark;
                async move {
                    let session = datafusion_bench::get_session_context();
                    datafusion_bench::make_object_store(&session, benchmark.data_url())?;
                    register_benchmark_tables(&session, benchmark, format).await?;
                    Ok((session, format))
                }
            },
            |query_idx, (session, format), query| {
                let plans = Arc::clone(&collected_plans);

                Box::pin(async move {
                    let timer = Instant::now();
                    let (batches, plan) = execute_query(session, query).await?;
                    let time = timer.elapsed();
                    let row_count = batches.iter().map(|batch| batch.num_rows()).sum::<usize>();

                    // Store plan for metrics (only store once per query/format combination)
                    if show_metrics {
                        let mut plans_mut = plans.lock();
                        // Only store if we don't already have this query/format combo
                        if !plans_mut
                            .iter()
                            .any(|(idx, f, _)| *idx == query_idx && *f == *format)
                        {
                            plans_mut.push((query_idx, *format, plan.clone()));
                        }
                    }

                    anyhow::Ok((row_count, Some(time), plan))
                })
            },
        )
        .await?;

    // Print metrics if requested
    if show_metrics {
        let plans = collected_plans.lock();
        print_metrics(plans.as_ref());
    }

    let benchmark_id = format!("datafusion-{}", benchmark.dataset_name());
    let writer = create_output_writer(&args.display_format, args.output_path, &benchmark_id)?;
    runner.export_to(&args.display_format, writer)?;

    Ok(())
}

async fn register_benchmark_tables<B: Benchmark + ?Sized>(
    session: &SessionContext,
    benchmark: &B,
    format: Format,
) -> anyhow::Result<()> {
    match format {
        Format::Arrow => register_arrow_tables(session, benchmark).await,
        _ => {
            let benchmark_base = benchmark.data_url().join(&format!("{}/", format.name()))?;
            let file_format = format_to_df_format(format);

            for table in benchmark.table_specs().iter() {
                let pattern = benchmark.pattern(table.name, format);
                let table_url = ListingTableUrl::try_new(benchmark_base.clone(), pattern)?;

                let mut config = ListingTableConfig::new(table_url).with_listing_options(
                    ListingOptions::new(file_format.clone())
                        .with_session_config_options(session.state().config()),
                );

                config = match table.schema.as_ref() {
                    Some(schema) => config.with_schema(Arc::new(schema.clone())),
                    None => config.infer_schema(&session.state()).await?,
                };

                let listing_table = Arc::new(ListingTable::try_new(config)?);

                session.register_table(table.name, listing_table)?;
            }

            Ok(())
        }
    }
}

/// Load Arrow IPC files into in-memory DataFusion tables.
async fn register_arrow_tables<B: Benchmark + ?Sized>(
    session: &SessionContext,
    benchmark: &B,
) -> anyhow::Result<()> {
    use datafusion::datasource::MemTable;

    let parquet_dir = benchmark
        .data_url()
        .to_file_path()
        .map_err(|_| anyhow::anyhow!("Arrow format requires local file path"))?
        .join(Format::Parquet.name());

    // Read all arrow files from the directory
    let data_files = std::fs::read_dir(&parquet_dir)?.collect::<Result<Vec<_>, _>>()?;

    for table in benchmark.table_specs().iter() {
        let pattern = benchmark.pattern(table.name, Format::Parquet);

        // Find files matching this table's pattern
        let matching_files: Vec<_> = data_files
            .iter()
            .filter(|entry| {
                let filename = entry.file_name();
                let filename_str = filename.to_str().unwrap_or("");
                match &pattern {
                    Some(p) => p.matches(filename_str),
                    None => filename_str == format!("{}.{}", table.name, Format::Parquet.ext()),
                }
            })
            .collect();

        // Load all matching files into memory
        let mut all_batches = Vec::new();
        let mut schema = None;

        for dir_entry in matching_files {
            let file = File::open(dir_entry.path()).await?;
            let mut reader = ParquetRecordBatchStreamBuilder::new(file).await?.build()?;
            if schema.is_none() {
                schema = Some(reader.schema()).cloned();
            }

            while let Some(batch) = reader.next().await {
                all_batches.push(batch?);
            }
        }

        if let Some(schema) = schema {
            let mem_table = MemTable::try_new(schema, vec![all_batches])?;
            session.register_table(table.name, Arc::new(mem_table))?;
        }
    }

    Ok(())
}

pub async fn execute_query(
    ctx: &SessionContext,
    query: &str,
) -> anyhow::Result<(Vec<RecordBatch>, Arc<dyn ExecutionPlan>)> {
    let df = ctx.sql(query).await?;

    let task_ctx = Arc::new(df.task_ctx());
    let plan = df.create_physical_plan().await?;
    let result = collect(plan.clone(), task_ctx).await?;

    Ok((result, plan))
}

/// Print Vortex metrics from execution plans.
fn print_metrics(plans: &[(usize, Format, Arc<dyn ExecutionPlan>)]) {
    for (query_idx, format, plan) in plans {
        let metric_sets = VortexMetricsFinder::find_all(plan.as_ref());
        if metric_sets.is_empty() {
            continue;
        }

        eprintln!("metrics for query={query_idx}, {format}:");
        for (scan_idx, metrics_set) in metric_sets.iter().enumerate() {
            eprintln!("  scan[{scan_idx}]:");
            for metric in metrics_set.clone().aggregate().sorted_for_display().iter() {
                eprintln!("    {metric}");
            }
        }
    }
}
