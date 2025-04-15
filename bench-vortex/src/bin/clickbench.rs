use std::cell::OnceCell;
use std::fs::{self};
use std::str::FromStr;
use std::time::{Duration, Instant};

use bench_vortex::clickbench::{self, Flavor, HITS_SCHEMA, clickbench_queries};
use bench_vortex::display::{DisplayFormat, RatioMode, print_measurements_json, render_table};
use bench_vortex::measurements::QueryMeasurement;
use bench_vortex::metrics::{MetricsSetExt, export_plan_spans};
use bench_vortex::{
    Engine, Format, IdempotentPath, default_env_filter, execute_query, feature_flagged_allocator,
    get_session_with_cache, make_object_store,
};
use clap::Parser;
use datafusion_physical_plan::display::DisplayableExecutionPlan;
use datafusion_physical_plan::execution_plan;
use indicatif::ProgressBar;
use log::warn;
use tokio::runtime::Builder;
use tracing::info_span;
use tracing_futures::Instrument;
use url::Url;
use vortex::error::{VortexExpect, vortex_panic};
use vortex_datafusion::persistent::metrics::VortexMetricsFinder;

feature_flagged_allocator!();

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, value_delimiter = ',', value_enum, default_values_t = vec![Engine::DataFusion])]
    engines: Vec<Engine>,
    #[arg(long)]
    duckdb_path: Option<std::path::PathBuf>,
    #[arg(short, long, default_value_t = 5)]
    iterations: usize,
    #[arg(short, long)]
    threads: Option<usize>,
    #[arg(
        long,
        value_delimiter = ',',
        value_enum,
        default_values_t = vec![Format::Parquet, Format::OnDiskVortex]
    )]
    formats: Vec<Format>,
    #[arg(long)]
    only_vortex: bool,
    #[arg(short, long)]
    verbose: bool,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    #[arg(long, default_value_t = false)]
    emit_plan: bool,
    #[arg(short, long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,
    #[arg(long, default_value_t = false)]
    emulate_object_store: bool,
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
    hide_metrics: bool,
}

struct DataFusionCtx {
    execution_plans: Vec<(usize, std::sync::Arc<dyn execution_plan::ExecutionPlan>)>,
    metrics: Vec<(
        usize,
        Format,
        Vec<datafusion_physical_plan::metrics::MetricsSet>,
    )>,

    session: datafusion::execution::context::SessionContext,
    emit_execution_plan: bool,
}

struct DuckDBCtx {
    duckdb_path: Option<std::path::PathBuf>,
}

enum EngineCtx {
    DataFusion(DataFusionCtx),
    DuckDB(DuckDBCtx),
}

impl EngineCtx {
    fn new_with_datafusion(
        session_ctx: datafusion::execution::context::SessionContext,
        emit_execution_plan: bool,
    ) -> Self {
        EngineCtx::DataFusion(DataFusionCtx {
            execution_plans: std::vec::Vec::new(),
            metrics: std::vec::Vec::new(),
            session: session_ctx,
            emit_execution_plan,
        })
    }

    fn new_with_duckdb(duckdb_path: Option<std::path::PathBuf>) -> Self {
        EngineCtx::DuckDB(DuckDBCtx { duckdb_path })
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

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

    if args.only_vortex {
        panic!("use `--formats vortex` instead of `--only-vortex`");
    }

    let tokio_runtime = || {
        match args.threads {
            Some(0) => panic!("Can't use 0 threads for runtime"),
            Some(1) => Builder::new_current_thread().enable_all().build(),
            Some(n) => Builder::new_multi_thread()
                .worker_threads(n)
                .enable_all()
                .build(),
            None => Builder::new_multi_thread().enable_all().build(),
        }
        .expect("Failed building the Runtime")
    };

    let queries = match &args.queries {
        None => clickbench_queries(),
        Some(queries) => clickbench_queries()
            .into_iter()
            .filter(|(q_idx, _)| queries.contains(q_idx))
            .collect(),
    };

    let base_url = data_source_base_url(&args.use_remote_data_dir, args.flavor)?;

    let progress_bar = if args.hide_progress_bar {
        ProgressBar::hidden()
    } else {
        ProgressBar::new((queries.len() * args.formats.len() * args.engines.len()) as u64)
    };

    let mut query_measurements = Vec::new();

    for engine in &args.engines {
        for file_format in &args.formats {
            let session_ctx = get_session_with_cache(args.emulate_object_store);

            // Register object store to the session.
            make_object_store(&session_ctx, &base_url).expect("Failed to make object store");

            let mut engine_ctx = match engine {
                Engine::DataFusion => EngineCtx::new_with_datafusion(session_ctx, args.emit_plan),
                Engine::DuckDB => EngineCtx::new_with_duckdb(args.duckdb_path.clone()),
                _ => unreachable!("engine not supported"),
            };

            let tokio_runtime = tokio_runtime();

            init_data_source(
                *file_format,
                &base_url,
                args.single_file,
                &engine_ctx,
                &tokio_runtime,
            )
            .expect("Failed to initialize data source");

            let bench_measurements = execute_queries(
                &queries,
                args.iterations,
                args.single_file,
                &tokio_runtime,
                *file_format,
                &base_url,
                &progress_bar,
                &mut engine_ctx,
            );

            if let EngineCtx::DataFusion(ref ctx) = engine_ctx {
                if args.export_spans {
                    if let Err(err) = tokio_runtime.block_on(async move {
                        export_plan_spans(*file_format, &ctx.execution_plans).await
                    }) {
                        warn!("failed to export spans {err}");
                    }
                }

                if !args.hide_metrics {
                    print_metrics(&ctx.metrics);
                }
            }

            query_measurements.extend(bench_measurements);
        }
    }

    print_results(
        &args.display_format,
        query_measurements,
        &args.formats,
        &args.engines,
    );

    Ok(())
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
    file_formats: &[Format],
    engines: &[Engine],
) {
    match display_format {
        DisplayFormat::Table => {
            render_table(query_measurements, file_formats, RatioMode::Time, engines).unwrap()
        }

        DisplayFormat::GhJson => print_measurements_json(query_measurements).unwrap(),
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
    tokio_runtime: &tokio::runtime::Runtime,
) -> anyhow::Result<()> {
    match file_format {
        Format::Parquet => match engine_ctx {
            EngineCtx::DataFusion(ctx) => clickbench::register_parquet_files(
                &ctx.session,
                "hits",
                base_url,
                &HITS_SCHEMA,
                single_file,
            )?,
            EngineCtx::DuckDB(_) => { /* nothing to do */ }
        },
        Format::OnDiskVortex => {
            tokio_runtime.block_on(async {
                if base_url.scheme() == "file" {
                    clickbench::convert_parquet_to_vortex(&base_url.to_file_path().unwrap())
                        .await
                        .unwrap_or_else(|err| panic!("init of {file_format} failed with: {err}"));
                }

                match engine_ctx {
                    EngineCtx::DataFusion(ctx) => {
                        clickbench::register_vortex_files(
                            ctx.session.clone(),
                            "hits",
                            base_url,
                            &HITS_SCHEMA,
                            single_file,
                        )
                        .unwrap_or_else(|err| panic!("init of {file_format} failed with: {err}"));
                    }

                    EngineCtx::DuckDB(_) => { /* nothing to do */ }
                }
            });
        }
        _ => vortex_panic!("Format {file_format} isn't supported on ClickBench"),
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
    single_file: bool,
    tokio_runtime: &tokio::runtime::Runtime,
    file_format: Format,
    base_url: &Url,
    progress_bar: &ProgressBar,
    engine_ctx: &mut EngineCtx,
) -> Vec<QueryMeasurement> {
    let mut query_measurements = Vec::default();

    for (query_idx, query_string) in queries {
        match engine_ctx {
            EngineCtx::DataFusion(ctx) => {
                let (fastest_run, execution_plan) = benchmark_datafusion_query(
                    *query_idx,
                    query_string,
                    iterations,
                    &ctx.session,
                    tokio_runtime,
                );

                ctx.execution_plans
                    .push((*query_idx, execution_plan.clone()));

                if ctx.emit_execution_plan {
                    write_execution_plan(*query_idx, file_format, &execution_plan);
                }

                ctx.metrics.push((
                    *query_idx,
                    file_format,
                    VortexMetricsFinder::find_all(execution_plan.as_ref()),
                ));

                query_measurements.push(QueryMeasurement {
                    query_idx: *query_idx,
                    engine: Engine::DataFusion,
                    storage: "nvme".to_owned(),
                    time: fastest_run,
                    format: file_format,
                    dataset: "clickbench".to_owned(),
                });
            }

            EngineCtx::DuckDB(args) => {
                let duckdb_path = duckdb_executable_path(&args.duckdb_path);

                let fastest_run = benchmark_duckdb_query(
                    *query_idx,
                    query_string,
                    iterations,
                    file_format,
                    base_url,
                    single_file,
                    &duckdb_path,
                );

                query_measurements.push(QueryMeasurement {
                    query_idx: *query_idx,
                    engine: Engine::DuckDB,
                    storage: "nvme".to_string(),
                    time: fastest_run,
                    format: file_format,
                    dataset: "clickbench".to_owned(),
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
fn benchmark_datafusion_query(
    query_idx: usize,
    query_string: &str,
    iterations: usize,
    context: &datafusion::execution::context::SessionContext,
    tokio_runtime: &tokio::runtime::Runtime,
) -> (Duration, std::sync::Arc<dyn execution_plan::ExecutionPlan>) {
    let execution_plan = OnceCell::new();

    let fastest_run =
        (0..iterations).fold(Duration::from_millis(u64::MAX), |fastest, iteration| {
            tokio_runtime.block_on(async {
                let (duration, plan) =
                    execute_datafusion_query(query_idx, query_string, iteration, context.clone())
                        .await
                        .unwrap_or_else(|err| panic!("query: {query_idx} failed with: {err}"));

                if execution_plan.get().is_none() {
                    execution_plan
                        .set(plan)
                        .expect("assign the execution plan only once");
                }

                fastest.min(duration)
            })
        });

    (
        fastest_run,
        execution_plan
            .into_inner()
            .expect("Execution plan must be set"),
    )
}

async fn execute_datafusion_query(
    query_idx: usize,
    query_string: &str,
    iteration: usize,
    session_context: datafusion::execution::context::SessionContext,
) -> anyhow::Result<(Duration, std::sync::Arc<dyn execution_plan::ExecutionPlan>)> {
    let query_string = query_string.to_owned();

    let (duration, execution_plan) = tokio::task::spawn(async move {
        let time_instant = Instant::now();
        let (_, execution_plan) = execute_query(&session_context, &query_string)
            .instrument(info_span!("execute_query", query_idx, iteration))
            .await
            .unwrap_or_else(|e| panic!("executing query {query_idx}: {e}"));

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
#[allow(clippy::let_and_return)]
fn benchmark_duckdb_query(
    query_idx: usize,
    query_string: &str,
    iterations: usize,
    file_format: Format,
    base_url: &Url,
    single_file: bool,
    duckdb_path: &std::path::Path,
) -> Duration {
    let fastest_run = (0..iterations).fold(Duration::from_millis(u64::MAX), |fastest, _| {
        let duration = execute_duckdb_query(
            query_string,
            base_url,
            file_format,
            single_file,
            duckdb_path,
        )
        .unwrap_or_else(|err| panic!("query: {query_idx} failed with: {err}"));

        fastest.min(duration)
    });

    fastest_run
}

fn execute_duckdb_query(
    query_string: &str,
    base_url: &Url,
    file_format: Format,
    single_file: bool,
    duckdb_path: &std::path::Path,
) -> anyhow::Result<Duration> {
    let extension = match file_format {
        Format::Parquet => "parquet",
        Format::OnDiskVortex => "vortex",
        other => vortex_panic!("Format {other} isn't supported on ClickBench"),
    };

    // Base path contains trailing /.
    let file_glob = if single_file {
        format!("{base_url}{extension}/hits.{extension}")
    } else {
        format!("{base_url}{extension}/*.{extension}")
    };

    let file_glob = file_glob.strip_prefix("file://").unwrap_or(&file_glob);
    let time_instant = Instant::now();
    let register_tables =
        format!("CREATE VIEW hits AS SELECT * FROM read_{extension}('{file_glob}')",);

    let output = std::process::Command::new(duckdb_path)
        .arg("-c")
        .arg(register_tables)
        .arg("-c")
        .arg(query_string)
        .output()?;

    // DuckDB does not return non-zero exit codes in case of failures.
    // Therefore, we need to additionally check whether stderr is set.
    if !output.status.success() || !output.stderr.is_empty() {
        anyhow::bail!(
            "DuckDB query failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    Ok(time_instant.elapsed())
}

fn duckdb_executable_path(
    user_supplied_path_flag: &Option<std::path::PathBuf>,
) -> std::path::PathBuf {
    let validate_path = |duckdb_path: &std::path::PathBuf| {
        if !duckdb_path.as_path().exists() {
            panic!(
                "failed to find duckdb executable at: {}",
                duckdb_path.display()
            );
        }
    };

    // User supplied path takes priority.
    if let Some(duckdb_path) = user_supplied_path_flag {
        validate_path(duckdb_path);
        return duckdb_path.to_owned();
    }

    // Try to find the 'vortex' top-level directory. This is preferred over logic along
    // the lines of `git rev-parse --show-toplevel`, as the repository uses submodules.
    let mut repo_root = None;
    let mut current_dir = std::env::current_dir().expect("failed to get current dir");

    while current_dir.file_name().is_some() {
        if current_dir.file_name().and_then(|name| name.to_str()) == Some("vortex") {
            repo_root = Some(current_dir.to_string_lossy().into_owned());
            break;
        }

        if !current_dir.pop() {
            break;
        }
    }

    let duckdb_path = std::path::PathBuf::from_str(&format!(
        "{}/duckdb-vortex/build/release/duckdb",
        repo_root.unwrap_or_default()
    ))
    .expect("failed to create DuckDB executable path");

    validate_path(&duckdb_path);

    duckdb_path
}

/// Writes execution plan details to files.
///
/// Creates 2 plan files for each query execution:
/// - A detailed plan with full structure
/// - A condensed plan with metrics and schema
fn write_execution_plan(
    query_idx: usize,
    format: Format,
    execution_plan: &std::sync::Arc<dyn execution_plan::ExecutionPlan>,
) {
    fs::write(
        format!("clickbench_{format}_q{query_idx:02}.plan"),
        format!("{:#?}", execution_plan),
    )
    .expect("Unable to write file");

    fs::write(
        format!("clickbench_{format}_q{query_idx:02}.short.plan"),
        format!(
            "{}",
            DisplayableExecutionPlan::with_full_metrics(execution_plan.as_ref())
                .set_show_schema(true)
                .set_show_statistics(true)
                .indent(true)
        ),
    )
    .expect("Unable to write file");
}
