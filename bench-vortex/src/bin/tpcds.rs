use std::path::PathBuf;

use anyhow::anyhow;
use bench_vortex::ddb::{DuckDBExecutor, register_tables};
use bench_vortex::df::write_execution_plan;
use bench_vortex::display::{DisplayFormat, print_measurements_json, render_table};
use bench_vortex::measurements::QueryMeasurement;
use bench_vortex::metrics::export_plan_spans;
use bench_vortex::tpcds::tpcds_queries;
use bench_vortex::tpch::duckdb::{DuckdbTpcOptions, TpcDataset, generate_tpc};
use bench_vortex::utils::TPCH_DATASET;
use bench_vortex::{
    BenchmarkDataset, Engine, Format, IdempotentPath, Target, ddb, default_env_filter, tpcds,
    vortex_panic,
};
use clap::{Parser, value_parser};
use datafusion_physical_plan::metrics::{Label, MetricsSet};
use indicatif::ProgressBar;
use itertools::Itertools;
use log::{info, warn};
use tempfile::tempdir;
use tpcds::load_datasets;
use url::Url;
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
}

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

    for format in formats {
        let opts = DuckdbTpcOptions::default()
            .with_scale_factor(1)
            .with_base_dir("tpcds".to_data_path())
            .with_dataset(TpcDataset::TpcDs)
            .with_format(format);
        generate_tpc(opts).expect("gen tpch-ds");
    }

    // Require trace guard lives until here
    #[cfg(feature = "tracing")]
    let _ = _trace_guard;
    Ok(())
}

#[allow(clippy::too_many_arguments)]
async fn bench_main(
    queries: Option<Vec<usize>>,
    iterations: usize,
    targets: Vec<Target>,
    display_format: DisplayFormat,
    url: Url,
    duckdb_path: &Option<PathBuf>,
) -> anyhow::Result<()> {
    info!(
        "Benchmarking against these targets: {}.",
        targets.iter().join(", ")
    );

    let query_count = queries.as_ref().map_or(99, |c| c.len());
    let progress = ProgressBar::new((query_count * targets.len()) as u64);
    let mut row_counts: Vec<(usize, Format, usize)> = Vec::new();
    let mut measurements = Vec::new();
    let mut metrics = MetricsSet::new();
    let tpch_queries: Vec<_> = tpcds_queries()
        .filter(|(query_idx, _)| {
            // Include query if:
            // 1. No specific queries were requested OR this query is in the requested list
            // 2. AND this query is not in the excluded list
            queries
                .as_ref()
                .is_none_or(|included| included.contains(query_idx))
        })
        .collect();

    assert!(!tpch_queries.is_empty(), "No queries to run");

    let duckdb_resolved_path = targets
        .iter()
        .any(|t| t.engine() == Engine::DuckDB)
        .then(|| ddb::build_and_get_executable_path(duckdb_path));

    for target in &targets {
        let engine = target.engine();
        let format = target.format();
        match engine {
            Engine::DataFusion => {
                let ctx = load_datasets(&url, format).await?;

                for (query_idx, sql_queries) in tpch_queries.clone() {
                    // Run benchmark as an async function
                    let (row_count, fastest_run, plan) =
                        benchmark_datafusion_query(query_idx, &sql_queries, iterations, &ctx).await;

                    row_counts.push((query_idx, format, row_count));

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
            // TODO(joe); ensure that files are downloaded before running duckdb.
            Engine::DuckDB => {
                let duckdb_path = duckdb_resolved_path.as_ref().vortex_expect("created above");
                let temp_dir = tempdir()?;
                let duckdb_file = temp_dir
                    .path()
                    .join(format!("duckdb-file-{}.db", format.name()));

                let executor = DuckDBExecutor::new(duckdb_path.clone(), duckdb_file);
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
