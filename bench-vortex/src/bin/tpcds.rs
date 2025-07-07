// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::{Write, stdout};
use std::path::PathBuf;

use bench_vortex::display::{DisplayFormat, print_measurements_json, render_table};
use bench_vortex::engines::{EngineCtx, benchmark_datafusion_query, benchmark_duckdb_query};
use bench_vortex::measurements::QueryMeasurement;
use bench_vortex::tpcds::tpcds_queries;
use bench_vortex::tpch::duckdb::{DuckdbTpcOptions, TpcDataset, generate_tpc};
use bench_vortex::tpch::load_datasets;
use bench_vortex::utils::new_tokio_runtime;
use bench_vortex::{
    BenchmarkDataset, Engine, IdempotentPath, Target, default_env_filter, vortex_panic,
};
use clap::{Parser, value_parser};
use indicatif::ProgressBar;
use itertools::Itertools;
use log::info;
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
    #[arg(short)]
    output_path: Option<PathBuf>,
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

    for format in formats {
        let opts = DuckdbTpcOptions::new("tpcds".to_data_path(), TpcDataset::TpcDs, format);
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
        1, // scale factor 1 for now
        args.display_format,
        url,
        &args.output_path,
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
    scale_factor: u32,
    display_format: DisplayFormat,
    url: Url,
    output_path: &Option<PathBuf>,
) -> anyhow::Result<()> {
    let dataset = BenchmarkDataset::TpcDS { scale_factor };
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
                if let EngineCtx::DuckDB(ctx) =
                    &EngineCtx::new_with_duckdb(dataset.clone(), format)?
                {
                    ctx.register_tables(&url, format, &dataset)?;

                    for (query_idx, sql_query) in &tpch_queries {
                        let (fastest_run, _row_count) =
                            benchmark_duckdb_query(*query_idx, sql_query, iterations, ctx);

                        let storage = bench_vortex::utils::url_scheme_to_storage(&url)?;

                        measurements.push(QueryMeasurement {
                            query_idx: *query_idx,
                            target: *target,
                            benchmark_dataset: dataset.clone(),
                            storage,
                            fastest_run,
                        });

                        progress.inc(1);
                    }
                } else {
                    return Err(anyhow::anyhow!("Expected DuckDB engine context"));
                }
            }
            Engine::DataFusion => {
                // TODO: add schemas for tpcds.
                let ctx = load_datasets(&url, format, &dataset, true).await?;

                for (query_idx, sql_queries) in tpch_queries.clone() {
                    let (fastest_run, _) = benchmark_datafusion_query(iterations, || async {
                        let (record_batches, _metrics) =
                            bench_vortex::df::execute_query(&ctx, &sql_queries)
                                .await
                                .unwrap_or_else(|err| {
                                    vortex_panic!("query: {query_idx} failed with: {err}")
                                });
                        let q_row_count =
                            record_batches.iter().map(|r| r.num_rows()).sum::<usize>();
                        q_row_count
                    })
                    .await;

                    let storage = bench_vortex::utils::url_scheme_to_storage(&url)?;

                    measurements.push(QueryMeasurement {
                        query_idx,
                        target: *target,
                        benchmark_dataset: dataset.clone(),
                        storage,
                        fastest_run,
                    });

                    progress.inc(1);
                }
            }
            _ => todo!(),
        }
    }

    progress.finish();

    let mut writer: Box<dyn Write> = if let Some(output_path) = output_path {
        Box::new(File::create(output_path)?)
    } else {
        let stdout = stdout();
        Box::new(stdout.lock())
    };

    match display_format {
        DisplayFormat::Table => {
            render_table(&mut writer, measurements, &targets)?;
        }
        DisplayFormat::GhJson => {
            print_measurements_json(&mut writer, measurements)?;
        }
    };
    Ok(())
}
