use std::path::{Path, PathBuf};

use bench_vortex::ddb::{DuckDBExecutor, register_tables};
use bench_vortex::display::{DisplayFormat, print_measurements_json, render_table};
use bench_vortex::measurements::QueryMeasurement;
use bench_vortex::tpcds::{benchmark_duckdb_query, tpcds_queries};
use bench_vortex::tpch::duckdb::{DuckdbTpcOptions, TpcDataset, generate_tpc};
use bench_vortex::utils::{TPCDS_DATASET, new_tokio_runtime};
use bench_vortex::{BenchmarkDataset, Engine, IdempotentPath, Target, ddb, default_env_filter};
use clap::{Parser, value_parser};
use indicatif::ProgressBar;
use itertools::Itertools;
use log::{info, warn};
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
    // Don't try to rebuild duckdb
    #[arg(long)]
    skip_rebuild: bool,
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

    let duckdb_resolved_path =
        ddb::build_and_get_executable_path(&args.duckdb_path, args.skip_rebuild);

    for format in formats {
        let opts = DuckdbTpcOptions::default()
            .with_scale_factor(1)
            .with_base_dir("tpcds".to_data_path())
            .with_dataset(TpcDataset::TpcDs)
            .with_duckdb_path(duckdb_resolved_path.clone())
            .with_format(format);
        generate_tpc(opts).expect("gen tpch-ds");
    }

    let url = Url::parse(
        ("file:".to_owned()
            + "tpcds"
                .to_data_path()
                .to_str()
                .vortex_expect("path should be utf8")
            + "/")
            .as_ref(),
    )?;

    let runtime = new_tokio_runtime(None);

    runtime.block_on(bench_main(
        args.queries,
        args.exclude_queries,
        args.iterations,
        1,
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
    scale_factor: usize,
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
                let duckdb_path = duckdb_resolved_path;
                let temp_dir = tempdir()?;
                let duckdb_file = temp_dir
                    .path()
                    .join(format!("duckdb-file-{}.db", format.name()));

                let url = url
                    .join(&format!("{scale_factor}/"))
                    .vortex_expect("cannot join scale factor");

                let executor = DuckDBExecutor::new(duckdb_path.to_owned(), duckdb_file);
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
            _ => {
                warn!("Engine {:?} not supported for TPC-H benchmarks", engine);
            }
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
