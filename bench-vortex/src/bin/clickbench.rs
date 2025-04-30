use std::cell::OnceCell;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use bench_vortex::clickbench::{Flavor, clickbench_queries};
use bench_vortex::ddb::{DuckDBExecutor, register_tables};
use bench_vortex::display::{DisplayFormat, print_measurements_json, render_table};
use bench_vortex::measurements::QueryMeasurement;
use bench_vortex::metrics::{MetricsSetExt, export_plan_spans};
use bench_vortex::utils::constants::{CLICKBENCH_DATASET, STORAGE_NVME};
use bench_vortex::utils::new_tokio_runtime;
use bench_vortex::{
    BenchmarkDataset, Engine, Format, IdempotentPath, Target, ddb, default_env_filter, df,
};
use clap::{Parser, value_parser};
use datafusion::prelude;
use datafusion_physical_plan::ExecutionPlan;
use indicatif::ProgressBar;
use itertools::Itertools;
use log::warn;
use prelude::SessionContext;
use tempfile::{TempDir, tempdir};
use tokio::runtime::Runtime;
use tracing::{debug, info_span};
use tracing_futures::Instrument;
use url::Url;
use vortex::error::{VortexExpect, vortex_panic};
use vortex_datafusion::persistent::metrics::VortexMetricsFinder;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Target),
        default_values = vec![
            "datafusion:parquet",
            "datafusion:vortex",
            "duckdb:parquet",
            "duckdb:vortex",
            "duckdb:duckdb"
        ]
    )]
    targets: Vec<Target>,
    #[arg(long)]
    duckdb_path: Option<PathBuf>,
    #[arg(short, long, default_value_t = 5)]
    iterations: usize,
    #[arg(short, long)]
    threads: Option<usize>,
    #[arg(short, long)]
    verbose: bool,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    #[arg(long, default_value_t = false)]
    emit_plan: bool,
    #[arg(short, long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,
    #[arg(long)]
    queries_file: Option<PathBuf>,
    #[arg(long, default_value_t = false)]
    disable_datafusion_cache: bool,
    #[arg(long)]
    export_spans: bool,
    #[arg(long, value_enum, default_value_t = Flavor::Partitioned)]
    flavor: Flavor,
    #[arg(long)]
    use_remote_data_dir: Option<String>,
    #[arg(long, default_value_t = false)]
    single_file: bool,
    #[arg(long, default_value_t = false)]
    hide_progress_bar: bool,
    #[arg(long, default_value_t = false)]
    show_metrics: bool,
}

struct DataFusionCtx {
    execution_plans: Vec<(usize, Arc<dyn ExecutionPlan>)>,
    metrics: Vec<(
        usize,
        Format,
        Vec<datafusion::physical_plan::metrics::MetricsSet>,
    )>,

    session: SessionContext,
    emit_plan: bool,
}

struct DuckDBCtx {
    duckdb_path: PathBuf,
    tmp_dir: TempDir,
}

impl DuckDBCtx {
    pub fn duckdb_file(&self, format: Format) -> PathBuf {
        self.tmp_dir
            .path()
            .to_path_buf()
            .join(format!("hits-{format}.db"))
    }
}

enum EngineCtx {
    DataFusion(DataFusionCtx),
    DuckDB(DuckDBCtx),
}

impl EngineCtx {
    fn new_with_datafusion(session_ctx: SessionContext, emit_plan: bool) -> Self {
        EngineCtx::DataFusion(DataFusionCtx {
            execution_plans: Vec::new(),
            metrics: Vec::new(),
            session: session_ctx,
            emit_plan,
        })
    }

    fn new_with_duckdb(duckdb_path: &Path) -> Self {
        EngineCtx::DuckDB(DuckDBCtx {
            duckdb_path: duckdb_path.to_path_buf(),
            tmp_dir: tempdir().vortex_expect("cannot open temp directory"),
        })
    }

    fn to_engine(&self) -> Engine {
        match &self {
            EngineCtx::DuckDB(_) => Engine::DuckDB,
            EngineCtx::DataFusion(_) => Engine::DataFusion,
        }
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let engines = args
        .targets
        .iter()
        .map(|t| t.engine())
        .unique()
        .collect_vec();
    let formats = args
        .targets
        .iter()
        .map(|t| t.format())
        .unique()
        .collect_vec();

    validate_args(&engines, &args);

    // Capture `RUST_LOG` configuration
    let filter = default_env_filter(args.verbose);

    #[cfg(not(feature = "tracing"))]
    bench_vortex::setup_logger(filter);

    // We need the guard to live to the end of the function, so can't create it in the if-block
    #[cfg(feature = "tracing")]
    let _trace_guard = {
        use std::io::IsTerminal;

        use tracing_subscriber::prelude::*;

        let (layer, _guard) = tracing_chrome::ChromeLayerBuilder::new()
            .include_args(true)
            .trace_style(tracing_chrome::TraceStyle::Async)
            .file("clickbench.trace.json")
            .build();

        let fmt_layer = tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .with_level(true)
            .with_line_number(true)
            .with_ansi(std::io::stderr().is_terminal());

        tracing_subscriber::registry()
            .with(filter)
            .with(layer)
            .with(fmt_layer)
            .init();
        _guard
    };

    let queries_filepath = args
        .queries_file
        .unwrap_or_else(|| Path::new(env!("CARGO_MANIFEST_DIR")).join("clickbench_queries.sql"));

    debug!(file = ?queries_filepath, "Reading queries from file");

    let queries = match &args.queries {
        None => clickbench_queries(queries_filepath),
        Some(queries) => clickbench_queries(queries_filepath)
            .into_iter()
            .filter(|(q_idx, _)| queries.contains(q_idx))
            .collect(),
    };

    let base_url = data_source_base_url(&args.use_remote_data_dir, args.flavor)?;

    let progress_bar = if args.hide_progress_bar {
        ProgressBar::hidden()
    } else {
        ProgressBar::new((queries.len() * formats.len() * engines.len()) as u64)
    };

    let mut query_measurements = Vec::new();

    let resolved_path = args
        .targets
        .iter()
        .any(|t| t.engine() == Engine::DuckDB)
        .then(|| ddb::build_and_get_executable_path(&args.duckdb_path));

    for target in args.targets.iter() {
        let engine = target.engine();
        let file_format = target.format();

        let mut engine_ctx = match engine {
            Engine::DataFusion => {
                let session_ctx = df::get_session_context(args.disable_datafusion_cache);
                // Register object store to the session.
                df::make_object_store(&session_ctx, &base_url)?;

                EngineCtx::new_with_datafusion(session_ctx, args.emit_plan)
            }
            Engine::DuckDB => EngineCtx::new_with_duckdb(
                resolved_path.as_ref().vortex_expect("path resolved above"),
            ),
            _ => unreachable!("engine not supported"),
        };

        let tokio_runtime = new_tokio_runtime(args.threads);

        init_data_source(
            file_format,
            &base_url,
            args.single_file,
            &engine_ctx,
            &tokio_runtime,
        )?;

        let bench_measurements = execute_queries(
            &queries,
            args.iterations,
            &tokio_runtime,
            file_format,
            &progress_bar,
            &mut engine_ctx,
        );

        if let EngineCtx::DataFusion(ref ctx) = engine_ctx {
            if args.export_spans {
                if let Err(err) = tokio_runtime.block_on(async move {
                    export_plan_spans(file_format, &ctx.execution_plans).await
                }) {
                    warn!("failed to export spans {err}");
                }
            }

            if args.show_metrics {
                print_metrics(&ctx.metrics);
            }
        }

        query_measurements.extend(bench_measurements);
    }

    print_results(&args.display_format, query_measurements, &args.targets)
}

fn validate_args(engines: &[Engine], args: &Args) {
    assert!(
        args.duckdb_path.is_none() || engines.contains(&Engine::DuckDB),
        "--duckdb-path is only valid when DuckDB engine is used"
    );

    if (args.emit_plan || args.export_spans || args.show_metrics || args.threads.is_some())
        && !engines.contains(&Engine::DataFusion)
    {
        vortex_panic!(
            "--emit-plan, --export-spans, --show_metrics, --threads are only valid if DataFusion is used"
        );
    }
}

fn print_metrics(
    metrics: &Vec<(
        usize,
        Format,
        Vec<datafusion::physical_plan::metrics::MetricsSet>,
    )>,
) {
    for (query_idx, file_format, metric_sets) in metrics {
        eprintln!("metrics for query={query_idx}, {file_format}:");
        for (query_idx, metrics_set) in metric_sets.iter().enumerate() {
            eprintln!("scan[{query_idx}]:");
            for metric in metrics_set
                .clone()
                .timestamps_removed()
                .aggregate()
                .sorted_for_display()
                .iter()
            {
                eprintln!("{metric}");
            }
        }
    }
}

fn print_results(
    display_format: &DisplayFormat,
    query_measurements: Vec<QueryMeasurement>,
    targets: &[Target],
) -> anyhow::Result<()> {
    match display_format {
        DisplayFormat::Table => render_table(query_measurements, targets),

        DisplayFormat::GhJson => print_measurements_json(query_measurements),
    }
}

/// Determines the URL location for benchmark data, either local or remote.
///
/// If `remote_data_dir` is None, data is downloaded to a local path and a file:// URL is returned.
/// Otherwise, the provided remote URL (s3://, gs://, etc.) is validated and returned.
fn data_source_base_url(remote_data_dir: &Option<String>, flavor: Flavor) -> anyhow::Result<Url> {
    match remote_data_dir {
        None => {
            let basepath = format!("clickbench_{}", flavor).to_data_path();
            let client = reqwest::blocking::Client::default();

            flavor.download(&client, basepath.as_path())?;
            Ok(Url::parse(&format!(
                "file:{}/",
                basepath.to_str().vortex_expect("path should be utf8")
            ))?)
        }
        Some(remote_data_dir) => {
            // e.g. "s3://vortex-bench-dev-eu/parquet/"
            if !remote_data_dir.ends_with("/") {
                log::warn!(
                    "Supply a --use-remote-data-dir argument which ends in a slash e.g. s3://vortex-bench-dev-eu/parquet/"
                );
            }
            log::info!(
                concat!(
                    "Assuming data already exists at this remote (e.g. S3, GCS) URL: {}.\n",
                    "If it does not, you should kill this command, locally generate the files (by running without\n",
                    "--use-remote-data-dir) and upload data/clickbench/ to some remote location.",
                ),
                remote_data_dir,
            );
            Ok(Url::parse(remote_data_dir)?)
        }
    }
}

/// Configures the data source format for benchmark queries based on the specified format and engine.
///
/// Parquet files are registered directly. Vortex files are created form Parquet files.
fn init_data_source(
    file_format: Format,
    base_url: &Url,
    single_file: bool,
    engine_ctx: &EngineCtx,
    tokio_runtime: &Runtime,
) -> anyhow::Result<()> {
    let dataset = BenchmarkDataset::ClickBench { single_file };

    if file_format == Format::OnDiskVortex && base_url.scheme() == "file" {
        let file_path = base_url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("invalid file URL: {}", base_url))?;
        tokio_runtime.block_on(bench_vortex::file::convert_parquet_to_vortex(
            &file_path, dataset,
        ))?
    }

    match engine_ctx {
        EngineCtx::DataFusion(ctx) => match file_format {
            Format::Parquet | Format::OnDiskVortex => {
                dataset.register_tables(&ctx.session, base_url, file_format)?
            }
            _ => {
                vortex_panic!(
                    "Engine {} Format {file_format} isn't supported on ClickBench",
                    engine_ctx.to_engine()
                )
            }
        },
        EngineCtx::DuckDB(ctx) => match file_format {
            Format::Parquet | Format::OnDiskVortex | Format::OnDiskDuckDB => register_tables(
                &DuckDBExecutor::new(ctx.duckdb_path.clone(), ctx.duckdb_file(file_format)),
                base_url,
                file_format,
                dataset,
            )?,
            _ => {
                vortex_panic!(
                    "Engine {} Format {file_format} isn't supported on ClickBench",
                    engine_ctx.to_engine()
                )
            }
        },
    }

    Ok(())
}

/// Executes all provided ClickBench queries with the specified engine and data format.
///
/// # Arguments
///
/// * `queries` - Query indices and their corresponding SQL strings
/// * `iterations` - Number of times to execute each query
/// * `single_file` - Whether to use a single file or multiple files for the dataset
/// * `tokio_runtime` - Tokio runtime
/// * `file_format` - Parquet, Vortex, etc.
/// * `base_url` - Base URL where the dataset is located
/// * `progress_bar` - Progress indicator for tracking query execution
/// * `engine_ctx` - DataFusion or DuckDB context
#[allow(clippy::too_many_arguments)]
fn execute_queries(
    queries: &[(usize, String)],
    iterations: usize,
    tokio_runtime: &Runtime,
    file_format: Format,
    progress_bar: &ProgressBar,
    engine_ctx: &mut EngineCtx,
) -> Vec<QueryMeasurement> {
    let mut query_measurements = Vec::default();

    for &(query_idx, ref query_string) in queries.iter() {
        match engine_ctx {
            EngineCtx::DataFusion(ctx) => {
                let (fastest_run, execution_plan) = benchmark_datafusion_query(
                    query_idx,
                    query_string,
                    iterations,
                    &ctx.session,
                    tokio_runtime,
                );

                ctx.execution_plans
                    .push((query_idx, execution_plan.clone()));

                if ctx.emit_plan {
                    df::write_execution_plan(
                        query_idx,
                        file_format,
                        CLICKBENCH_DATASET,
                        execution_plan.as_ref(),
                    );
                }

                ctx.metrics.push((
                    query_idx,
                    file_format,
                    VortexMetricsFinder::find_all(execution_plan.as_ref()),
                ));

                query_measurements.push(QueryMeasurement {
                    query_idx,
                    target: Target::new(Engine::DataFusion, file_format),
                    storage: STORAGE_NVME.to_owned(),
                    fastest_run,
                    dataset: CLICKBENCH_DATASET.to_owned(),
                });
            }

            EngineCtx::DuckDB(args) => {
                let fastest_run = benchmark_duckdb_query(
                    query_idx,
                    query_string,
                    iterations,
                    &DuckDBExecutor::new(args.duckdb_path.clone(), args.duckdb_file(file_format)),
                );

                query_measurements.push(QueryMeasurement {
                    query_idx,
                    target: Target::new(Engine::DuckDB, file_format),
                    storage: STORAGE_NVME.to_owned(),
                    fastest_run,
                    dataset: CLICKBENCH_DATASET.to_owned(),
                });
            }
        };

        progress_bar.inc(1);
    }

    query_measurements
}

/// Executes a single ClickBench query using DataFusion.
///
/// # Returns
///
/// - The duration of the fastest execution
/// - The execution plan used for the query
#[allow(clippy::unwrap_used)]
fn benchmark_datafusion_query(
    query_idx: usize,
    query_string: &str,
    iterations: usize,
    context: &SessionContext,
    tokio_runtime: &Runtime,
) -> (Duration, Arc<dyn ExecutionPlan>) {
    let execution_plan = OnceCell::new();

    let fastest_run =
        (0..iterations).fold(Duration::from_millis(u64::MAX), |fastest, iteration| {
            tokio_runtime.block_on(async {
                let (duration, plan) =
                    execute_datafusion_query(query_idx, query_string, iteration, context.clone())
                        .await
                        .unwrap_or_else(|err| {
                            vortex_panic!("query: {query_idx} failed with: {err}")
                        });

                if execution_plan.get().is_none() {
                    execution_plan.set(plan).unwrap();
                }

                fastest.min(duration)
            })
        });

    (
        fastest_run,
        execution_plan
            .into_inner()
            .vortex_expect("Execution plan must be set"),
    )
}

async fn execute_datafusion_query(
    query_idx: usize,
    query_string: &str,
    iteration: usize,
    session_context: SessionContext,
) -> anyhow::Result<(Duration, Arc<dyn ExecutionPlan>)> {
    let query_string = query_string.to_owned();

    let (duration, execution_plan) = tokio::task::spawn(async move {
        let time_instant = Instant::now();
        let (_, execution_plan) = df::execute_query(&session_context, &query_string)
            .instrument(info_span!("execute_query", query_idx, iteration))
            .await
            .unwrap_or_else(|e| vortex_panic!("executing query {query_idx}: {e}"));

        (time_instant.elapsed(), execution_plan)
    })
    .await?;

    Ok((duration, execution_plan))
}

/// Executes a single ClickBench query using DuckDB.
///
/// # Returns
///
/// The duration of the fastest execution
fn benchmark_duckdb_query(
    query_idx: usize,
    query_string: &str,
    iterations: usize,
    duckdb_executor: &DuckDBExecutor,
) -> Duration {
    (0..iterations).fold(Duration::from_millis(u64::MAX), |fastest, _| {
        let duration = ddb::execute_clickbench_query(query_string, duckdb_executor)
            .unwrap_or_else(|err| vortex_panic!("query: {query_idx} failed with: {err}"));

        fastest.min(duration)
    })
}
