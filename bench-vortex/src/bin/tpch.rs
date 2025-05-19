use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use anyhow::anyhow;
use bench_vortex::ddb::{DuckDBExecutor, register_tables};
use bench_vortex::df::write_execution_plan;
use bench_vortex::display::{DisplayFormat, print_measurements_json, render_table};
use bench_vortex::measurements::QueryMeasurement;
use bench_vortex::metrics::{MetricsSetExt, export_plan_spans};
use bench_vortex::tpch::duckdb::{DuckdbTpcOptions, TpcDataset, generate_tpc};
use bench_vortex::tpch::{
    EXPECTED_ROW_COUNTS_SF1, EXPECTED_ROW_COUNTS_SF10, TPC_H_ROW_COUNT_ARRAY_LENGTH, load_datasets,
    run_tpch_query, tpch_queries,
};
use bench_vortex::utils::constants::TPCH_DATASET;
use bench_vortex::utils::new_tokio_runtime;
use bench_vortex::{
    BenchmarkDataset, Engine, Format, IdempotentPath, Target, ddb, default_env_filter, vortex_panic,
};
use clap::{Parser, ValueEnum, value_parser};
use datafusion::execution::context::SessionContext;
use datafusion::physical_plan::execution_plan::ExecutionPlan;
use datafusion::physical_plan::metrics::{Label, MetricsSet};
use indicatif::ProgressBar;
use itertools::Itertools;
use log::{info, warn};
use tempfile::tempdir;
use url::Url;
use vortex::aliases::hash_map::HashMap;
use vortex::error::VortexExpect;
use vortex_datafusion::persistent::metrics::VortexMetricsFinder;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
struct Args {
    #[arg(long, value_delimiter = ',', value_parser = value_parser!(Target),
        default_values = vec![
            "datafusion:parquet",
            "datafusion:vortex",
            "datafusion:arrow",
            "duckdb:parquet",
            "duckdb:vortex",
            "duckdb:duckdb"
        ]
    )]
    targets: Vec<Target>,
    #[arg(long)]
    duckdb_path: Option<PathBuf>,
    #[arg(short, long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,
    #[arg(short, long, value_delimiter = ',')]
    exclude_queries: Option<Vec<usize>>,
    #[arg(short, long)]
    threads: Option<usize>,
    #[arg(long)]
    use_remote_data_dir: Option<String>,
    #[arg(short, long, default_value_t = 10)]
    iterations: usize,
    #[arg(long, default_value_t = 1)]
    scale_factor: u8,
    #[arg(short)]
    verbose: bool,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    #[arg(long, default_value_t = false)]
    disable_datafusion_cache: bool,
    #[arg(long, default_value_t, value_enum)]
    data_generator: DataGenerator,
    #[arg(long)]
    all_metrics: bool,
    #[arg(long)]
    export_spans: bool,
    #[arg(long, default_value_t = false)]
    emit_plan: bool,
    #[arg(long)]
    skip_duckdb_build: bool,
}

#[derive(ValueEnum, Default, Clone, Debug, PartialEq, Eq)]
pub enum DataGenerator {
    #[default]
    Dbgen,
    Duckdb,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let engines = args.targets.iter().map(|t| t.engine()).collect_vec();

    validate_args(&engines, &args);

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
            .file("tpch.trace.json")
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

    let formats = args.targets.iter().map(|t| t.format()).collect_vec();

    let runtime = new_tokio_runtime(args.threads);

    let duckdb_resolved_path = ddb::duckdb_executable_path(&args.duckdb_path);
    if args.duckdb_path.is_none() && !args.skip_duckdb_build {
        ddb::build_vortex_duckdb();
    }

    let url = match args.use_remote_data_dir {
        None => {
            for format in formats {
                // Arrow uses csv
                let format = if format == Format::Arrow {
                    Format::Csv
                } else {
                    format
                };
                let opts = DuckdbTpcOptions::new("tpch".to_data_path(), TpcDataset::TpcH, format)
                    .with_duckdb_path(duckdb_resolved_path.clone());
                generate_tpc(opts)?;
            }

            let data_dir = "tpch".to_data_path();
            let data_dir = data_dir.to_str().vortex_expect("path must be utf8");

            info!("Using existing or generating new files located at {data_dir}.");
            Url::parse(format!("file:{data_dir}/{}/", args.scale_factor).as_ref())?
        }
        Some(tpch_benchmark_remote_data_dir) => {
            // e.g. "s3://vortex-bench-dev-eu/parquet/"
            //
            // The trailing slash is significant!
            //
            // The folder must already be populated with data!
            if !tpch_benchmark_remote_data_dir.ends_with("/") {
                warn!(
                    "Supply a --use-remote-data-dir argument which ends in a slash e.g. s3://vortex-bench-dev-eu/parquet/"
                );
            }
            info!(
                concat!(
                    "Assuming data already exists at this remote (e.g. S3, GCS) URL: {}.\n",
                    "If it does not, you should kill this command, locally generate the files (by running without\n",
                    "--use-remote-data-dir) and upload data/tpch/1/ to some remote location.",
                ),
                tpch_benchmark_remote_data_dir,
            );
            Url::parse(&tpch_benchmark_remote_data_dir)?
        }
    };

    runtime.block_on(bench_main(
        args.queries,
        args.exclude_queries,
        args.iterations,
        args.targets,
        args.display_format,
        args.disable_datafusion_cache,
        args.scale_factor,
        url,
        args.all_metrics,
        args.export_spans,
        args.emit_plan,
        duckdb_resolved_path,
    ))
}

/// Verify row counts against expected values for TPC-H queries
///
/// Returns true if there are any row count mismatches.
fn verify_row_counts(
    row_counts: &[(usize, Format, usize)],
    expected_row_counts: [usize; TPC_H_ROW_COUNT_ARRAY_LENGTH],
    queries: &Option<Vec<usize>>,
    exclude_queries: &Option<Vec<usize>>,
) -> bool {
    let format_row_counts =
        row_counts
            .iter()
            .fold(HashMap::new(), |mut acc, &(idx, format, row_count)| {
                acc.entry(format)
                    .or_insert_with(|| vec![0; TPC_H_ROW_COUNT_ARRAY_LENGTH])[idx] = row_count;
                acc
            });

    let is_query_included = |idx: &usize| {
        queries.as_ref().is_none_or(|q| q.contains(idx))
            && exclude_queries
                .as_ref()
                .is_none_or(|excluded| !excluded.contains(idx))
    };

    let mut mismatched = false;
    for (format, row_counts) in format_row_counts {
        row_counts
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| is_query_included(idx))
            .for_each(|(idx, actual_row_count)| {
                if actual_row_count != expected_row_counts[idx] {
                    if idx == 15 && actual_row_count == 0 {
                        warn!(
                            "*IGNORING* mismatched row count {} instead of {} for format {:?} because Query 15 is flaky. See: https://github.com/vortex-data/vortex/issues/2395",
                            actual_row_count,
                            expected_row_counts[idx],
                            format,
                        );
                    } else  {
                        warn!(
                            "Mismatched row count {} instead of {} in query {} for format {:?}",
                            actual_row_count,
                            expected_row_counts[idx],
                            idx,
                            format,
                        );
                        mismatched = true;
                    }
                }
            });
    }

    mismatched
}

fn benchmark_duckdb_query(
    query_idx: usize,
    queries: &[String],
    iterations: usize,
    duckdb_executor: &DuckDBExecutor,
) -> Duration {
    (0..iterations).fold(Duration::from_millis(u64::MAX), |fastest, _| {
        let duration = ddb::execute_tpch_query(queries, duckdb_executor)
            .unwrap_or_else(|err| vortex_panic!("query: {query_idx} failed with: {err}"));

        fastest.min(duration)
    })
}

async fn benchmark_datafusion_query(
    query_idx: usize,
    query_string: &[String],
    iterations: usize,
    context: &SessionContext,
) -> (usize, Duration, Arc<dyn ExecutionPlan>) {
    let mut row_count = usize::MAX;
    let mut fastest_run = Duration::from_millis(u64::MAX);
    let mut plan_result = None;

    for _ in 0..iterations {
        let start = Instant::now();
        let (q_row_count, plan) = run_tpch_query(context, query_string, query_idx).await;
        let elapsed = start.elapsed();
        row_count = q_row_count;

        if plan_result.is_none() {
            plan_result = Some(plan.clone());
        }

        fastest_run = fastest_run.min(elapsed);
    }

    (
        row_count,
        fastest_run,
        plan_result.vortex_expect("Execution plan must be set"),
    )
}

#[allow(clippy::too_many_arguments)]
async fn bench_main(
    queries: Option<Vec<usize>>,
    exclude_queries: Option<Vec<usize>>,
    iterations: usize,
    targets: Vec<Target>,
    display_format: DisplayFormat,
    disable_datafusion_cache: bool,
    scale_factor: u8,
    url: Url,
    display_all_metrics: bool,
    export_spans: bool,
    emit_plan: bool,
    duckdb_resolved_path: PathBuf,
) -> anyhow::Result<()> {
    let expected_row_counts = if scale_factor == 1 {
        EXPECTED_ROW_COUNTS_SF1
    } else if scale_factor == 10 {
        EXPECTED_ROW_COUNTS_SF10
    } else {
        vortex_panic!(
            "Scale factor {} not supported due to lack of expected row counts.",
            scale_factor
        );
    };

    info!(
        "Benchmarking against these targets: {}.",
        targets.iter().join(", ")
    );

    let query_count = queries.as_ref().map_or(22, |c| c.len());
    let progress = ProgressBar::new((query_count * targets.len()) as u64);
    let mut row_counts: Vec<(usize, Format, usize)> = Vec::new();
    let mut measurements = Vec::new();
    let mut metrics = MetricsSet::new();
    let tpch_queries: Vec<_> = tpch_queries()
        .filter(|(query_idx, _)| {
            // Include query if:
            // 1. No specific queries were requested OR this query is in the requested list
            // 2. AND this query is not in the excluded list
            queries
                .as_ref()
                .is_none_or(|included| included.contains(query_idx))
                && exclude_queries
                    .as_ref()
                    .is_none_or(|excluded| !excluded.contains(query_idx))
        })
        .collect();

    assert!(!tpch_queries.is_empty(), "No queries to run");

    for target in &targets {
        let engine = target.engine();
        let format = target.format();
        match engine {
            Engine::DataFusion => {
                let ctx = load_datasets(
                    &url,
                    format,
                    BenchmarkDataset::TpcH,
                    disable_datafusion_cache,
                )
                .await?;

                let mut plans = Vec::new();

                for (query_idx, sql_queries) in tpch_queries.clone() {
                    // Run benchmark as an async function
                    let (row_count, fastest_run, plan) =
                        benchmark_datafusion_query(query_idx, &sql_queries, iterations, &ctx).await;

                    row_counts.push((query_idx, format, row_count));

                    // Gather metrics.
                    for (idx, metrics_set) in VortexMetricsFinder::find_all(plan.as_ref())
                        .into_iter()
                        .enumerate()
                    {
                        metrics.merge_all_with_label(
                            metrics_set,
                            &[
                                Label::new("query_idx", query_idx.to_string()),
                                Label::new("vortex_exec_idx", idx.to_string()),
                            ],
                        );
                    }

                    if emit_plan {
                        write_execution_plan(query_idx, format, TPCH_DATASET, plan.as_ref());
                    }

                    plans.push((query_idx, plan.clone()));

                    let storage = bench_vortex::utils::url_scheme_to_storage(&url)?;

                    measurements.push(QueryMeasurement {
                        query_idx,
                        target: *target,
                        storage,
                        fastest_run,
                        dataset: TPCH_DATASET.to_owned(),
                    });

                    progress.inc(1);
                }

                if export_spans {
                    if let Err(e) = export_plan_spans(format, &plans).await {
                        warn!("failed to export spans {e}");
                    }
                }
            }
            // TODO(joe); ensure that files are downloaded before running duckdb.
            Engine::DuckDB => {
                let temp_dir = tempdir()?;
                let duckdb_file = temp_dir
                    .path()
                    .join(format!("duckdb-file-{}.db", format.name()));

                let executor = DuckDBExecutor::new(duckdb_resolved_path.clone(), duckdb_file);
                register_tables(&executor, &url, format, BenchmarkDataset::TpcH)?;

                for (query_idx, sql_queries) in tpch_queries.clone() {
                    let fastest_run =
                        benchmark_duckdb_query(query_idx, &sql_queries, iterations, &executor);

                    let storage = bench_vortex::utils::url_scheme_to_storage(&url)?;

                    measurements.push(QueryMeasurement {
                        query_idx,
                        target: *target,
                        storage,
                        fastest_run,
                        dataset: TPCH_DATASET.to_owned(),
                    });

                    progress.inc(1);
                }
            }
            _ => {
                warn!("Engine {:?} not supported for TPC-H benchmarks", engine);
            }
        }
    }

    progress.finish();

    match display_format {
        DisplayFormat::Table => {
            if !display_all_metrics {
                metrics = metrics.aggregate();
            }
            for m in metrics.timestamps_removed().sorted_for_display().iter() {
                println!("{}", m);
            }
            render_table(measurements, &targets)?;
        }
        DisplayFormat::GhJson => {
            print_measurements_json(measurements)?;
        }
    }

    if verify_row_counts(&row_counts, expected_row_counts, &queries, &exclude_queries) {
        Err(anyhow!("Mismatched row counts. See logs for details."))
    } else {
        anyhow::Ok(())
    }
}

fn validate_args(engines: &[Engine], args: &Args) {
    assert!(
        args.duckdb_path.is_none() || engines.contains(&Engine::DuckDB),
        "--duckdb-path is only valid if DuckDB is used"
    );

    if (args.all_metrics || args.export_spans || args.emit_plan || args.threads.is_some())
        && !engines.contains(&Engine::DataFusion)
    {
        vortex_panic!(
            "--all-metrics, --emit-plan, --threads, --export-spans are only valid if DataFusion is used"
        );
    }
}
