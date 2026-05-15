// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::Parser;
use clap::value_parser;
use custom_labels::asynchronous::Label;
use datafusion::arrow::array::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::common::runtime::set_join_set_tracer;
use datafusion::datasource::listing::ListingOptions;
use datafusion::datasource::listing::ListingTable;
use datafusion::datasource::listing::ListingTableConfig;
use datafusion::datasource::listing::ListingTableUrl;
use datafusion::parquet::arrow::ParquetRecordBatchStreamBuilder;
use datafusion::prelude::SessionContext;
use datafusion_bench::format_to_df_format;
use datafusion_bench::metrics::MetricsSetExt;
use datafusion_bench::tracer::get_labelset_from_global;
use datafusion_bench::tracer::get_static_tracer;
use datafusion_bench::tracer::set_labels;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::collect;
use futures::StreamExt;
use parking_lot::Mutex;
use tokio::fs::File;
use vortex::io::filesystem::FileSystemRef;
use vortex::scan::DataSourceRef;
use vortex_bench::Benchmark;
use vortex_bench::BenchmarkArg;
use vortex_bench::CompactionStrategy;
use vortex_bench::Engine;
use vortex_bench::Format;
use vortex_bench::Opt;
use vortex_bench::Opts;
use vortex_bench::SESSION;
use vortex_bench::conversions::convert_parquet_directory_to_vortex;
use vortex_bench::create_benchmark;
use vortex_bench::create_output_writer;
use vortex_bench::display::DisplayFormat;
use vortex_bench::runner::BenchmarkMode;
use vortex_bench::runner::BenchmarkQueryResult;
use vortex_bench::runner::SqlBenchmarkRunner;
use vortex_bench::runner::filter_queries;
use vortex_bench::setup_logging_and_tracing;
use vortex_bench::v3;
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

    /// Additionally write v3 JSONL records to this path. See
    /// `benchmarks-website/planning/02-contracts.md`.
    #[arg(long)]
    gh_json_v3: Option<PathBuf>,

    #[arg(long, default_value_t = false)]
    show_metrics: bool,

    #[arg(long, default_value_t = false)]
    hide_progress_bar: bool,

    #[arg(long, default_value_t = false)]
    emit_plan: bool,

    #[arg(long, default_value_t = false)]
    track_memory: bool,

    #[arg(long, default_value = "unknown")]
    runner: String,

    #[arg(long, default_value_t = false)]
    explain: bool,

    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Format))]
    formats: Vec<Format>,

    #[arg(long = "opt", value_delimiter = ',', value_parser = value_parser!(Opt))]
    options: Vec<Opt>,
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    let opts = Opts::from(args.options);

    set_join_set_tracer(get_static_tracer())?;
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
                    convert_parquet_directory_to_vortex(&base_path, CompactionStrategy::Default)
                        .await?;
                }
                Format::VortexCompact => {
                    convert_parquet_directory_to_vortex(&base_path, CompactionStrategy::Compact)
                        .await?;
                }
                _ => {}
            }
        }
    }

    let benchmark_name = benchmark.dataset().to_string();

    let mut runner = SqlBenchmarkRunner::new(
        &*benchmark,
        Engine::DataFusion,
        args.runner.clone(),
        args.formats.clone(),
        args.track_memory,
        args.hide_progress_bar,
    )?;

    // Collect execution plans for metrics if show_metrics is enabled
    // Structure: (query_idx, format, execution_plan)
    #[expect(clippy::type_complexity)]
    let collected_plans: Arc<Mutex<Vec<(usize, Format, Arc<dyn ExecutionPlan>)>>> =
        Arc::new(Mutex::new(Vec::new()));
    let show_metrics = args.show_metrics;

    let mode = if args.explain {
        BenchmarkMode::Explain
    } else {
        BenchmarkMode::Run {
            iterations: args.iterations,
        }
    };

    runner
        .run_all_async(
            &filtered_queries,
            mode,
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

                let labelset = set_labels(benchmark_name.clone(), query_idx, *format);

                Box::pin(
                    async move {
                        let timer = Instant::now();
                        let (batches, plan) = execute_query(session, query)
                            .with_labelset(get_labelset_from_global())
                            .await?;
                        let time = timer.elapsed();

                        // Store plan for metrics (only store once per query/format combination)
                        if show_metrics {
                            let mut plans_mut = plans.lock();
                            // Only store if we don't already have this query/format combo
                            if !plans_mut
                                .iter()
                                .any(|(idx, f, _)| *idx == query_idx && *f == *format)
                            {
                                plans_mut.push((query_idx, *format, Arc::clone(&plan)));
                            }
                        }

                        anyhow::Ok((Some(time), DataFusionQueryResult(batches)))
                    }
                    .with_labelset(labelset),
                )
            },
        )
        .await?;

    if !args.explain {
        // Print metrics if requested
        if show_metrics {
            let plans = collected_plans.lock();
            print_metrics(plans.as_ref());
        }

        if let Some(path) = args.gh_json_v3.as_ref() {
            v3::write_jsonl_to_path(path, &runner.v3_records())?;
        }

        let benchmark_id = format!("datafusion-{}", benchmark.dataset_name());
        let writer = create_output_writer(&args.display_format, args.output_path, &benchmark_id)?;
        runner.export_to(&args.display_format, writer)?;
    }

    Ok(())
}

fn use_scan_api() -> bool {
    std::env::var("VORTEX_USE_SCAN_API").is_ok_and(|v| v == "1")
}

async fn register_benchmark_tables<B: Benchmark + ?Sized>(
    session: &SessionContext,
    benchmark: &B,
    format: Format,
) -> anyhow::Result<()> {
    match format {
        Format::Arrow => register_arrow_tables(session, benchmark).await,
        _ if use_scan_api() && matches!(format, Format::OnDiskVortex | Format::VortexCompact) => {
            register_v2_tables(session, benchmark, format).await
        }
        _ => {
            let benchmark_base = benchmark.data_url().join(&format!("{}/", format.name()))?;
            let file_format = format_to_df_format(format);

            for table in benchmark.table_specs().iter() {
                let pattern = benchmark.pattern(table.name, format);
                let table_url = ListingTableUrl::try_new(benchmark_base.clone(), pattern)?;

                let mut listing_options = ListingOptions::new(Arc::clone(&file_format))
                    .with_session_config_options(session.state().config());
                if benchmark.dataset_name() == "polarsignals" && format == Format::Parquet {
                    // Work around a DataFusion bug (fixed in 53.0.0) where the
                    // constant-column optimization extracts ScalarValues using
                    // the statistic scalar type, which may not match the table
                    // column type.
                    // See: https://github.com/apache/datafusion/pull/20042
                    // TODO(asubiotto): Remove this after the datafusion 53
                    // upgrade.
                    listing_options = listing_options.with_collect_stat(false);
                }
                let mut config =
                    ListingTableConfig::new(table_url).with_listing_options(listing_options);

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

/// Register tables using the V2 `VortexTable` + `MultiFileDataSource` path.
async fn register_v2_tables<B: Benchmark + ?Sized>(
    session: &SessionContext,
    benchmark: &B,
    format: Format,
) -> anyhow::Result<()> {
    use vortex::file::multi::MultiFileDataSource;
    use vortex::io::object_store::ObjectStoreFileSystem;
    use vortex::io::session::RuntimeSessionExt;
    use vortex::scan::DataSource as _;
    use vortex_datafusion::v2::VortexTable;

    let benchmark_base = benchmark.data_url().join(&format!("{}/", format.name()))?;

    for table in benchmark.table_specs().iter() {
        let pattern = benchmark.pattern(table.name, format);
        let table_url = ListingTableUrl::try_new(benchmark_base.clone(), pattern.clone())?;
        let store = session
            .state()
            .runtime_env()
            .object_store(table_url.object_store())?;

        let fs: FileSystemRef = Arc::new(ObjectStoreFileSystem::new(
            Arc::clone(&store),
            SESSION.handle(),
        ));
        let base_prefix = benchmark_base.path().trim_start_matches('/').to_string();
        let fs = fs.with_prefix(base_prefix);

        let glob_pattern = match &pattern {
            Some(p) => p.as_str().to_string(),
            None => format!("*.{}", format.ext()),
        };

        let multi_ds = MultiFileDataSource::new(SESSION.clone())
            .with_glob(glob_pattern, Some(fs))
            .build()
            .await?;

        let arrow_schema = Arc::new(multi_ds.dtype().to_arrow_schema()?);
        let data_source: DataSourceRef = Arc::new(multi_ds);

        let table_provider = Arc::new(VortexTable::new(data_source, SESSION.clone(), arrow_schema));
        session.register_table(table.name, table_provider)?;
    }

    Ok(())
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

/// Wrapper around DataFusion record batches implementing `BenchmarkQueryResult`.
pub struct DataFusionQueryResult(pub Vec<RecordBatch>);

impl BenchmarkQueryResult for DataFusionQueryResult {
    fn row_count(&self) -> usize {
        self.0.iter().map(|batch| batch.num_rows()).sum()
    }

    fn display(self) -> String {
        pretty_format_batches(&self.0)
            .map(|d| d.to_string())
            .unwrap_or_else(|e| format!("<error: {e}>"))
    }
}

pub async fn execute_query(
    ctx: &SessionContext,
    query: &str,
) -> anyhow::Result<(Vec<RecordBatch>, Arc<dyn ExecutionPlan>)> {
    let df = ctx
        .sql(query)
        .with_labelset(get_labelset_from_global())
        .await?;

    let task_ctx = Arc::new(df.task_ctx());
    let plan = df
        .create_physical_plan()
        .with_labelset(get_labelset_from_global())
        .await?;
    let result = collect(Arc::clone(&plan), task_ctx)
        .with_labelset(get_labelset_from_global())
        .await?;

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
            eprintln!("\tscan[{scan_idx}]:");
            for metric in metrics_set.aggregate().sorted_for_display().iter() {
                eprintln!("\t\t{metric}");
            }
        }
    }
}
