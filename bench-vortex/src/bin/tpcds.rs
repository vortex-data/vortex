use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use bench_vortex::ddb::{DuckDBExecutor, register_tables};
use bench_vortex::display::{DisplayFormat, print_measurements_json, render_table};
use bench_vortex::measurements::QueryMeasurement;
use bench_vortex::tpcds::{benchmark_duckdb_query, run_datafusion_tpcds_query, tpcds_queries};
use bench_vortex::tpch::duckdb::{DuckdbTpcOptions, TpcDataset, generate_tpc};
use bench_vortex::tpch::load_datasets;
use bench_vortex::utils::{TPCDS_DATASET, TPCH_DATASET, new_tokio_runtime};
use bench_vortex::{BenchmarkDataset, Engine, IdempotentPath, Target, ddb, default_env_filter};
use clap::{Parser, value_parser};
use datafusion::prelude::SessionContext;
use indicatif::ProgressBar;
use itertools::Itertools;
use log::info;
use tempfile::tempdir;
use url::Url;
use vortex::error::VortexExpect;

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
    #[arg(short, long, default_value_t = 10)]
    iterations: usize,
    #[arg(short)]
    verbose: bool,
    #[arg(short, long, default_value_t, value_enum)]
    display_format: DisplayFormat,
    #[arg(long, default_value_t = false)]
    disable_datafusion_cache: bool,
    #[arg(long)]
    export_spans: bool,
    #[arg(long, default_value_t = false)]
    emit_plan: bool,
    #[arg(long)]
    skip_duckdb_build: bool,
}

#[allow(clippy::expect_used)]
fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let filter = default_env_filter(args.verbose);
    #[cfg(not(feature = "tracing"))]
    bench_vortex::setup_logger(filter);

    #[cfg(feature = "tracing")]
    let _trace_guard = {
        use std::io::IsTerminal;

        use tracing_subscriber::prelude::*;

        let (layer, _guard) = tracing_chrome::ChromeLayerBuilder::new()
            .include_args(true)
            .trace_style(tracing_chrome::TraceStyle::Async)
            .file("tpcds.trace.json")
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

    let formats = args
        .targets
        .iter()
        .map(|t| t.format())
        .unique()
        .collect_vec();

    let duckdb_resolved_path = ddb::duckdb_executable_path(&args.duckdb_path);
    if args.duckdb_path.is_none() && !args.skip_duckdb_build {
        ddb::build_vortex_duckdb();
    }

    for format in formats {
        let opts = DuckdbTpcOptions::new("tpcds".to_data_path(), TpcDataset::TpcDs, format)
            .with_duckdb_path(duckdb_resolved_path.clone());
        generate_tpc(opts).expect("gen tpch-ds");
    }

    let url = Url::parse(
        format!(
            "file:{}/{}/",
            "tpcds"
                .to_data_path()
                .to_str()
                .vortex_expect("path must be utf8"),
            // scale factor 1
            1
        )
        .as_ref(),
    )?;

    let runtime = new_tokio_runtime(None);

    runtime.block_on(bench_main(
        args.queries,
        args.exclude_queries,
        args.iterations,
        args.targets,
        args.display_format,
        url,
        &duckdb_resolved_path,
    ))?;

    // Require trace guard lives until here
    #[cfg(feature = "tracing")]
    let _ = _trace_guard;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn bench_main(
    queries: Option<Vec<usize>>,
    exclude_queries: Option<Vec<usize>>,
    iterations: usize,
    targets: Vec<Target>,
    display_format: DisplayFormat,
    url: Url,
    duckdb_resolved_path: &Path,
) -> anyhow::Result<()> {
    info!(
        "Benchmarking against these targets: {}.",
        targets.iter().join(", ")
    );

    let mut measurements: Vec<QueryMeasurement> = Vec::new();
    let tpch_queries: Vec<_> = tpcds_queries()
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

    let progress = ProgressBar::new((tpch_queries.len() * targets.len()) as u64);

    for target in &targets {
        let engine = target.engine();
        let format = target.format();
        match engine {
            // TODO(joe): support datafusion
            Engine::DuckDB => {
                let temp_dir = tempdir()?;
                let duckdb_file = temp_dir
                    .path()
                    .join(format!("duckdb-file-{}.db", format.name()));

                let executor = DuckDBExecutor::new(duckdb_resolved_path, duckdb_file);
                register_tables(&executor, &url, format, BenchmarkDataset::TpcDS)?;

                for (query_idx, sql_query) in tpch_queries.clone() {
                    let fastest_run =
                        benchmark_duckdb_query(query_idx, &sql_query, iterations, &executor);

                    let storage = bench_vortex::utils::url_scheme_to_storage(&url)?;

                    measurements.push(QueryMeasurement {
                        query_idx,
                        target: *target,
                        storage,
                        fastest_run,
                        dataset: TPCDS_DATASET.to_owned(),
                    });

                    progress.inc(1);
                }
            }
            Engine::DataFusion => {
                // TODO: add schemas for tpcds.
                let ctx = load_datasets(&url, format, BenchmarkDataset::TpcDS, true).await?;

                for (query_idx, sql_queries) in tpch_queries.clone() {
                    // Run benchmark as an async function
                    let fastest_run =
                        benchmark_datafusion_query(&sql_queries, iterations, &ctx).await;

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
            _ => todo!(),
        }
    }

    progress.finish();

    match display_format {
        DisplayFormat::Table => {
            render_table(measurements, &targets)?;
        }
        DisplayFormat::GhJson => {
            print_measurements_json(measurements)?;
        }
    };
    Ok(())
}

async fn benchmark_datafusion_query(
    query_string: &str,
    iterations: usize,
    context: &SessionContext,
) -> Duration {
    let mut fastest_run = Duration::from_millis(u64::MAX);

    for _ in 0..iterations {
        let start = Instant::now();
        // TODO(joe): add row count
        let _q_row_count = run_datafusion_tpcds_query(context, query_string).await;
        let elapsed = start.elapsed();

        fastest_run = fastest_run.min(elapsed);
    }

    fastest_run
}
