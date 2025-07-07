// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};

use bench_vortex::clickbench::{Flavor, clickbench_queries};
use bench_vortex::display::{DisplayFormat, print_measurements_json, render_table};
use bench_vortex::engines::{EngineCtx, benchmark_datafusion_query, benchmark_duckdb_query};
use bench_vortex::measurements::QueryMeasurement;
use bench_vortex::metrics::{MetricsSetExt, export_plan_spans};
use bench_vortex::utils::constants::{CLICKBENCH_DATASET, STORAGE_NVME};
use bench_vortex::utils::new_tokio_runtime;
use bench_vortex::{
    BenchmarkDataset, Engine, Format, IdempotentPath, Target, default_env_filter, df,
};
use clap::{Parser, value_parser};
use indicatif::ProgressBar;
use io::stdout;
use itertools::Itertools;
use log::warn;
use tokio::runtime::Runtime;
use tracing::debug;
use url::Url;
use vortex::error::{VortexExpect, vortex_panic};
use vortex_datafusion::metrics::VortexMetricsFinder;

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
    #[arg(short)]
    output_path: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let engines = args
        .targets
        .iter()
        .map(|t| t.engine())
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
            .with_writer(io::stderr)
            .with_level(true)
            .with_line_number(true)
            .with_ansi(io::stderr().is_terminal());

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
        ProgressBar::new((queries.len() * args.targets.len()) as u64)
    };

    let mut query_measurements = Vec::new();

    for target in args.targets.iter() {
        let engine = target.engine();
        let format = target.format();
        let dataset = BenchmarkDataset::ClickBench {
            single_file: args.single_file,
            flavor: args.flavor,
        };

        let mut engine_ctx = match engine {
            Engine::DataFusion => {
                let session_ctx = df::get_session_context(args.disable_datafusion_cache);
                // Register object store to the session.
                df::make_object_store(&session_ctx, &base_url)?;

                EngineCtx::new_with_datafusion(session_ctx, args.emit_plan)
            }
            Engine::DuckDB => EngineCtx::new_with_duckdb(dataset.clone(), format)?,
            _ => unreachable!("engine not supported"),
        };

        let tokio_runtime = new_tokio_runtime(args.threads);

        tokio_runtime.block_on(init_data_source(format, &base_url, &dataset, &engine_ctx))?;

        let bench_measurements = execute_queries(
            &queries,
            args.iterations,
            &tokio_runtime,
            format,
            dataset,
            &progress_bar,
            &mut engine_ctx,
        );

        if let EngineCtx::DataFusion(ref ctx) = engine_ctx {
            if args.export_spans {
                if let Err(err) = tokio_runtime
                    .block_on(async move { export_plan_spans(format, &ctx.execution_plans).await })
                {
                    warn!("failed to export spans {err}");
                }
            }

            if args.show_metrics {
                print_metrics(&ctx.metrics);
            }
        }

        query_measurements.extend(bench_measurements);
    }

    print_results(
        &args.display_format,
        query_measurements,
        &args.targets,
        &args.output_path,
    )
}

fn validate_args(engines: &[Engine], args: &Args) {
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
    file_path: &Option<PathBuf>,
) -> anyhow::Result<()> {
    let mut writer: Box<dyn Write> = if let Some(file_path) = file_path {
        Box::new(File::create(file_path)?)
    } else {
        let stdout = stdout();
        Box::new(stdout.lock())
    };
    match display_format {
        DisplayFormat::Table => render_table(&mut writer, query_measurements, targets),

        DisplayFormat::GhJson => print_measurements_json(&mut writer, query_measurements),
    }
}

/// Determines the URL location for benchmark data, either local or remote.
///
/// If `remote_data_dir` is None, data is downloaded to a local path and a file:// URL is returned.
/// Otherwise, the provided remote URL (s3://, gs://, etc.) is validated and returned.
fn data_source_base_url(remote_data_dir: &Option<String>, flavor: Flavor) -> anyhow::Result<Url> {
    match remote_data_dir {
        None => {
            let basepath = format!("clickbench_{flavor}").to_data_path();
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
async fn init_data_source(
    file_format: Format,
    base_url: &Url,
    dataset: &BenchmarkDataset,
    engine_ctx: &EngineCtx,
) -> anyhow::Result<()> {
    if file_format == Format::OnDiskVortex && base_url.scheme() == "file" {
        let file_path = base_url
            .to_file_path()
            .map_err(|_| anyhow::anyhow!("invalid file URL: {}", base_url))?;
        bench_vortex::file::convert_parquet_to_vortex(&file_path, dataset).await?
    }

    match engine_ctx {
        EngineCtx::DataFusion(ctx) => match file_format {
            Format::Parquet | Format::OnDiskVortex => {
                dataset
                    .register_tables(&ctx.session, base_url, file_format)
                    .await?
            }
            _ => {
                vortex_panic!(
                    "Engine {} Format {file_format} isn't supported on ClickBench",
                    engine_ctx.to_engine()
                )
            }
        },
        EngineCtx::DuckDB(ctx) => match file_format {
            Format::Parquet | Format::OnDiskVortex | Format::OnDiskDuckDB => {
                ctx.register_tables(base_url, file_format, dataset)?;
            }
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
/// * `tokio_runtime` - Tokio runtime
/// * `file_format` - Parquet, Vortex, etc.
/// * `progress_bar` - Progress indicator for tracking query execution
/// * `engine_ctx` - DataFusion or DuckDB context
#[allow(clippy::too_many_arguments)]
fn execute_queries(
    queries: &[(usize, String)],
    iterations: usize,
    tokio_runtime: &Runtime,
    file_format: Format,
    dataset: BenchmarkDataset,
    progress_bar: &ProgressBar,
    engine_ctx: &mut EngineCtx,
) -> Vec<QueryMeasurement> {
    let mut query_measurements = Vec::default();

    const REFERENCE_ROW_COUNTS: [usize; 43] = [
        1, 1, 1, 1, 1, 1, 1, 18, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 4, 1, 10, 10, 10, 10,
        10, 10, 25, 25, 1, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    ];

    for &(query_idx, ref query_string) in queries.iter() {
        match engine_ctx {
            EngineCtx::DataFusion(ctx) => {
                let (fastest_run, (execution_plan, row_count)) = tokio_runtime.block_on(async {
                    benchmark_datafusion_query(iterations, || async {
                        let (batches, plan) = df::execute_query(&ctx.session, query_string)
                            .await
                            .unwrap_or_else(|err| {
                                vortex_panic!("query: {query_idx} failed with: {err}")
                            });
                        let row_count: usize = batches.iter().map(|batch| batch.num_rows()).sum();
                        (plan, row_count)
                    })
                    .await
                });

                assert_eq!(
                    row_count, REFERENCE_ROW_COUNTS[query_idx],
                    "Error: Row count mismatch for query idx {query_idx} - datafusion:{file_format}",
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
                    benchmark_dataset: dataset.clone(),
                    storage: STORAGE_NVME.to_owned(),
                    fastest_run,
                });
            }
            EngineCtx::DuckDB(ctx) => {
                let (fastest_run, row_count) =
                    benchmark_duckdb_query(query_idx, query_string, iterations, ctx);

                assert_eq!(
                    row_count, REFERENCE_ROW_COUNTS[query_idx],
                    "Error: Row count mismatch for query idx {query_idx} - duckdb:{file_format}",
                );

                query_measurements.push(QueryMeasurement {
                    query_idx,
                    target: Target::new(Engine::DuckDB, file_format),
                    benchmark_dataset: dataset.clone(),
                    storage: STORAGE_NVME.to_owned(),
                    fastest_run,
                });
            }
        };

        progress_bar.inc(1);
    }

    query_measurements
}
