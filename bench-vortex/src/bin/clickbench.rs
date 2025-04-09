use std::cell::OnceCell;
use std::fs::{self};
use std::time::{Duration, Instant};

use bench_vortex::clickbench::{self, Flavor, HITS_SCHEMA, clickbench_queries};
use bench_vortex::display::{DisplayFormat, RatioMode, print_measurements_json, render_table};
use bench_vortex::measurements::QueryMeasurement;
use bench_vortex::metrics::{MetricsSetExt, export_plan_spans};
use bench_vortex::{
    Format, IdempotentPath, default_env_filter, execute_query, feature_flagged_allocator,
    get_session_with_cache, make_object_store,
};
use clap::{Parser, ValueEnum};
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

#[derive(ValueEnum, Clone, Copy, Debug, Hash, Default, PartialEq, Eq)]
enum Engine {
    #[default]
    #[clap(name = "datafusion")]
    DataFusion,
    #[clap(name = "duckdb")]
    DuckDB,
}

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, value_enum, default_value_t = Engine::DataFusion)]
    engine: Engine,
    #[arg(short, long, default_value_t = 5)]
    iterations: usize,
    #[arg(short, long)]
    threads: Option<usize>,
    #[arg(long, value_delimiter = ',', value_enum, default_values_t = vec![Format::Parquet, Format::OnDiskVortex])]
    formats: Vec<Format>,
    #[arg(long)]
    only_vortex: bool,
    #[arg(short, long)]
    verbose: bool,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
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

    let runtime = match args.threads {
        Some(0) => panic!("Can't use 0 threads for runtime"),
        Some(1) => Builder::new_current_thread().enable_all().build(),
        Some(n) => Builder::new_multi_thread()
            .worker_threads(n)
            .enable_all()
            .build(),
        None => Builder::new_multi_thread().enable_all().build(),
    }
    .expect("Failed building the Runtime");

    let queries = match &args.queries {
        None => clickbench_queries(),
        Some(queries) => clickbench_queries()
            .into_iter()
            .filter(|(q_idx, _)| queries.contains(q_idx))
            .collect(),
    };

    let base_url = data_source_base_url(&args.use_remote_data_dir, args.flavor)?;
    let progress_bar = ProgressBar::new((queries.len() * args.formats.len()) as u64);
    let mut query_measurements = Vec::default();
    let mut metrics = Vec::default();

    for file_format in &args.formats {
        let session_context = get_session_with_cache(args.emulate_object_store);
        // Register object store to the session.
        make_object_store(&session_context, &base_url)?;
        let mut execution_plans = Vec::default();

        init_data_source(
            *file_format,
            args.engine,
            &session_context,
            &base_url,
            args.single_file,
            &runtime,
        )?;

        execute_queries(
            &queries,
            args.engine,
            args.iterations,
            args.single_file,
            &runtime,
            *file_format,
            &base_url,
            &progress_bar,
            &mut query_measurements,
            &session_context,
            &mut execution_plans,
            &mut metrics,
        );

        if args.export_spans {
            if let Err(e) = runtime
                .block_on(async move { export_plan_spans(*file_format, execution_plans).await })
            {
                warn!("failed to export spans {e}");
            }
        }
    }

    match args.display_format {
        DisplayFormat::Table => {
            for (query_idx, file_format, metric_sets) in metrics {
                println!("metrics for query={query_idx}, {file_format}:");
                for (query_idx, metrics_set) in metric_sets.into_iter().enumerate() {
                    println!("scan[{query_idx}]:");
                    for metric in metrics_set
                        .timestamps_removed()
                        .aggregate()
                        .sorted_for_display()
                        .iter()
                    {
                        println!("{metric}");
                    }
                }
            }
            render_table(query_measurements, &args.formats, RatioMode::Time).unwrap()
        }

        DisplayFormat::GhJson => print_measurements_json(query_measurements).unwrap(),
    }

    Ok(())
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
                    "Supply a --remote-data-dir argument which ends in a slash e.g. s3://vortex-bench-dev-eu/parquet/"
                );
            }
            log::info!(
                concat!(
                    "Assuming data already exists at this remote (e.g. S3, GCS) URL: {}.\n",
                    "If it does not, you should kill this command, locally generate the files (by running without\n",
                    "--remote-data-dir) and upload data/clickbench/ to some remote location.",
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
    engine: Engine,
    context: &datafusion::execution::context::SessionContext,
    url: &Url,
    single_file: bool,
    runtime: &tokio::runtime::Runtime,
) -> anyhow::Result<()> {
    match file_format {
        Format::Parquet => match engine {
            Engine::DataFusion => {
                clickbench::register_parquet_files(context, "hits", url, &HITS_SCHEMA, single_file)?
            }
            Engine::DuckDB => {
                vortex_panic!("{file_format} x {engine:?} not supported")
            }
        },
        Format::OnDiskVortex => {
            runtime.block_on(async {
                if url.scheme() == "file" {
                    clickbench::convert_parquet_to_vortex(&url.to_file_path().unwrap())
                        .await
                        .unwrap_or_else(|err| panic!("init of {file_format} failed with: {err}"));
                }

                match engine {
                    Engine::DataFusion => {
                        clickbench::register_vortex_files(
                            context.clone(),
                            "hits",
                            url,
                            &HITS_SCHEMA,
                            single_file,
                        )
                        .unwrap_or_else(|err| panic!("init of {file_format} failed with: {err}"));
                    }

                    Engine::DuckDB => {
                        vortex_panic!("{file_format} x {engine:?} not supported")
                    }
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
/// * `engine` - DataFusion or DuckDB
/// * `iterations` - Number of times to execute each query
/// * `single_file` - Whether to use a single file or multiple files for the dataset
/// * `runtime` - Tokio runtime
/// * `file_format` - Parquet, Vortex, etc.
/// * `base_url` - Base URL where the dataset is located
/// * `progress_bar` - Progress indicator for tracking query execution
/// * `query_measurements` - Vector to store query performance measurements
/// * `session_context` - DataFusion session context
/// * `execution_plans` - Vector to store execution plans
/// * `metrics` - Vector to store metrics
#[allow(clippy::too_many_arguments)]
fn execute_queries(
    queries: &[(usize, String)],
    engine: Engine,
    iterations: usize,
    single_file: bool,
    runtime: &tokio::runtime::Runtime,
    file_format: Format,
    base_url: &Url,
    progress_bar: &ProgressBar,
    query_measurements: &mut Vec<QueryMeasurement>,
    session_context: &datafusion::execution::context::SessionContext,
    execution_plans: &mut Vec<(usize, std::sync::Arc<dyn execution_plan::ExecutionPlan>)>,
    metrics: &mut Vec<(
        usize,
        Format,
        Vec<datafusion_physical_plan::metrics::MetricsSet>,
    )>,
) {
    for (query_idx, query_string) in queries {
        match engine {
            Engine::DataFusion => {
                let (fastest_result, execution_plan) = benchmark_datafusion_query(
                    *query_idx,
                    query_string,
                    iterations,
                    session_context,
                    runtime,
                );

                execution_plans.push((*query_idx, execution_plan.clone()));
                write_execution_plan(*query_idx, file_format, &execution_plan);

                metrics.push((
                    *query_idx,
                    file_format,
                    VortexMetricsFinder::find_all(execution_plan.as_ref()),
                ));

                query_measurements.push(QueryMeasurement {
                    query_idx: *query_idx,
                    storage: "nvme".to_string(),
                    time: fastest_result,
                    format: file_format,
                    dataset: "clickbench".to_string(),
                });
            }

            Engine::DuckDB => {
                let _fastest_run = benchmark_duckdb_query(
                    *query_idx,
                    query_string,
                    iterations,
                    runtime,
                    file_format,
                    base_url,
                    single_file,
                );
            }
        };

        progress_bar.inc(1);
    }
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
    runtime: &tokio::runtime::Runtime,
) -> (Duration, std::sync::Arc<dyn execution_plan::ExecutionPlan>) {
    let execution_plan = OnceCell::new();

    let fastest_run =
        (0..iterations).fold(Duration::from_millis(u64::MAX), |fastest, iteration| {
            runtime.block_on(async {
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
    let time_instant = Instant::now();
    let query_string = query_string.to_owned();

    let execution_plan = tokio::task::spawn(async move {
        let (_, execution_plan) = execute_query(&session_context, &query_string)
            .instrument(info_span!("execute_query", query_idx, iteration))
            .await
            .unwrap_or_else(|e| panic!("executing query {query_idx}: {e}"));

        execution_plan
    })
    .await?;

    Ok((time_instant.elapsed(), execution_plan))
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
    runtime: &tokio::runtime::Runtime,
    file_format: Format,
    url: &Url,
    single_file: bool,
) -> Duration {
    let fastest_run = (0..iterations).fold(Duration::from_millis(u64::MAX), |fastest, _| {
        runtime.block_on(async {
            let duration = execute_duckdb_query(
                query_idx,
                query_string,
                &std::path::PathBuf::from(url.as_str()),
                file_format,
                single_file,
            )
            .await
            .unwrap_or_else(|err| panic!("query: {query_idx} failed with: {err}"));

            fastest.min(duration)
        })
    });

    fastest_run
}

async fn execute_duckdb_query(
    query_idx: usize,
    query_string: &str,
    data_path: &std::path::Path,
    file_format: Format,
    single_file: bool,
) -> anyhow::Result<Duration> {
    let query_file = tempfile::tempdir()?
        .path()
        .join(format!("query_{query_idx}.sql"));

    fs::write(&query_file, query_string)?;

    let extension = match file_format {
        Format::Parquet => "parquet",
        Format::OnDiskVortex => "vortex",
        other => vortex_panic!("Format {other} isn't supported on ClickBench"),
    };

    let file_glob = if single_file {
        format!(
            "{}/{extension}/hits.{extension}",
            data_path.to_string_lossy()
        )
    } else {
        format!("{}/{extension}/*.{extension}", data_path.to_string_lossy())
    };

    let time_instant = Instant::now();

    // TODO: complete duckdb setup
    let output = tokio::process::Command::new("duckdb")
        // Create a temporary database in RAM.
        .arg(":memory:")
        .arg("-c")
        .arg(format!(
            "CREATE VIEW hits AS SELECT * FROM read_{extension}('{file_glob}');",
        ))
        .arg("-c")
        .arg(format!(".read {}", query_file.to_string_lossy()))
        .output()
        .await?;

    if !output.status.success() {
        anyhow::bail!(
            "DuckDB query failed: {}",
            String::from_utf8_lossy(&output.stderr)
        );
    }

    Ok(time_instant.elapsed())
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
        format!("clickbench_{format}_q{query_idx:02}.plan",),
        format!("{:#?}", execution_plan),
    )
    .expect("Unable to write file");

    fs::write(
        format!("clickbench_{format}_q{query_idx:02}.short.plan",),
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
