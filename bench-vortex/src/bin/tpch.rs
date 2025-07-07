// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::io::{Write, stdout};
use std::path::{Path, PathBuf};
use std::{env, fs};

use anyhow::anyhow;
use bench_vortex::df::write_execution_plan;
use bench_vortex::display::{DisplayFormat, print_measurements_json, render_table};
use bench_vortex::engines::{EngineCtx, benchmark_datafusion_query, benchmark_duckdb_query, ddb};
use bench_vortex::measurements::QueryMeasurement;
use bench_vortex::metrics::{MetricsSetExt, export_plan_spans};
use bench_vortex::tpch::duckdb::{DuckdbTpcOptions, TpcDataset, generate_tpc};
use bench_vortex::tpch::{
    EXPECTED_ROW_COUNTS_SF1, EXPECTED_ROW_COUNTS_SF10, load_datasets, run_tpch_query, tpch_queries,
};
use bench_vortex::utils::new_tokio_runtime;
use bench_vortex::{
    BenchmarkDataset, Engine, Format, IdempotentPath, Target, default_env_filter, vortex_panic,
};
use clap::{Parser, ValueEnum, value_parser};
use datafusion::physical_plan::metrics::{Label, MetricsSet};
use indicatif::ProgressBar;
use itertools::Itertools;
use log::{info, warn};
use similar::{ChangeTag, TextDiff};
use url::Url;
use vortex::error::VortexExpect;
use vortex_datafusion::metrics::VortexMetricsFinder;

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
    #[arg(short, long)]
    threads: Option<usize>,
    #[arg(long)]
    use_remote_data_dir: Option<String>,
    #[arg(short, long, default_value_t = 10)]
    iterations: usize,
    #[arg(long, default_value_t = 1, value_parser=validate_scale_factor)]
    scale_factor: u32,
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
    #[arg(short)]
    output_path: Option<PathBuf>,
}

fn validate_scale_factor(val: &str) -> Result<u32, String> {
    match val.parse::<u32>() {
        Ok(n) if [1, 10, 100, 1000].contains(&n) => Ok(n),
        _ => Err(String::from(
            "Value must be a scale factor of 1, 10, 100 or 1000",
        )),
    }
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
                    .with_scale_factor(args.scale_factor);
                generate_tpc(opts)?;
            }

            let data_dir = "tpch".to_data_path();
            let data_dir = data_dir.to_str().vortex_expect("path must be utf8");

            info!("Using existing or generating new files located at {data_dir}.");
            Url::parse(format!("file:{data_dir}/{}/", args.scale_factor).as_ref())?
        }
        Some(tpch_benchmark_remote_data_dir) => {
            // e.g. "s3://vortex-bench-dev-eu/parquet/"
            // The trailing slash is significant!
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
        &args.output_path,
    ))
}

#[allow(clippy::too_many_arguments)]
async fn bench_main(
    queries: Option<Vec<usize>>,
    exclude_queries: Option<Vec<usize>>,
    iterations: usize,
    targets: Vec<Target>,
    display_format: DisplayFormat,
    disable_datafusion_cache: bool,
    scale_factor: u32,
    url: Url,
    display_all_metrics: bool,
    export_spans: bool,
    emit_plan: bool,
    output_path: &Option<PathBuf>,
) -> anyhow::Result<()> {
    let dataset = BenchmarkDataset::TpcH { scale_factor };
    let expected_row_counts = if scale_factor == 1 {
        Some(EXPECTED_ROW_COUNTS_SF1)
    } else if scale_factor == 10 {
        Some(EXPECTED_ROW_COUNTS_SF10)
    } else {
        warn!(
            "Scale factor {} not supported due to lack of expected row counts.",
            scale_factor
        );
        None
    };

    info!(
        "Benchmarking against these targets: {}.",
        targets.iter().join(", ")
    );

    let query_count = queries.as_ref().map_or(22, |c| c.len());
    let progress = ProgressBar::new((query_count * targets.len()) as u64);
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
                let ctx = load_datasets(&url, format, &dataset, disable_datafusion_cache).await?;

                let mut plans = Vec::new();

                for (query_idx, sql_query) in tpch_queries.clone() {
                    let (runs, (row_count, plan)) =
                        benchmark_datafusion_query(iterations, || async {
                            run_tpch_query(&ctx, &sql_query).await
                        })
                        .await;

                    if let Some(expected_row_counts) = &expected_row_counts {
                        assert_eq!(
                            row_count, expected_row_counts[query_idx],
                            "Error: Row count mismatch for query idx {query_idx} - {engine}:{format}",
                        );
                    }

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
                        write_execution_plan(query_idx, format, dataset.name(), plan.as_ref());
                    }

                    plans.push((query_idx, plan.clone()));

                    let storage = bench_vortex::utils::url_scheme_to_storage(&url)?;

                    measurements.push(QueryMeasurement {
                        query_idx,
                        target: *target,
                        benchmark_dataset: dataset.clone(),
                        storage,
                        runs,
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
                let engine_ctx = EngineCtx::new_with_duckdb(dataset.clone(), format)?;

                if let EngineCtx::DuckDB(ctx) = &engine_ctx {
                    ctx.register_tables(&url, format, &dataset)?;

                    for (query_idx, sql_query) in tpch_queries.clone() {
                        let (runs, row_count) =
                            benchmark_duckdb_query(query_idx, &sql_query, iterations, ctx);

                        if let Some(expected_row_counts) = &expected_row_counts {
                            assert_eq!(
                                row_count, expected_row_counts[query_idx],
                                "Error: Row count mismatch for query idx {query_idx} - {engine}:{format}",
                            );
                        }

                        let storage = bench_vortex::utils::url_scheme_to_storage(&url)?;

                        measurements.push(QueryMeasurement {
                            query_idx,
                            target: *target,
                            benchmark_dataset: dataset.clone(),
                            storage,
                            runs,
                        });

                        progress.inc(1);
                    }
                } else {
                    return Err(anyhow::anyhow!("Expected DuckDB engine context"));
                }
            }
            _ => {
                warn!("Engine {engine:?} not supported for TPC-H benchmarks");
            }
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
            if !display_all_metrics {
                metrics = metrics.aggregate();
            }
            for m in metrics.timestamps_removed().sorted_for_display().iter() {
                println!("{m}");
            }
            render_table(&mut writer, measurements, &targets)?;
        }
        DisplayFormat::GhJson => {
            print_measurements_json(&mut writer, measurements)?;
        }
    }

    // The CI env var is defined by Github Actions.
    // https://docs.github.com/en/actions/writing-workflows/choosing-what-your-workflow-does/store-information-in-variables#default-environment-variables
    if targets
        .iter()
        .any(|t| t.engine() == Engine::DuckDB && t.format() == Format::OnDiskVortex)
        && env::var("CI").is_ok()
    {
        verify_duckdb_tpch_results(&url, scale_factor, queries)?;
    }

    anyhow::Ok(())
}

fn verify_duckdb_tpch_results(
    url: &Url,
    scale_factor: u32,
    queries: Option<Vec<usize>>,
) -> anyhow::Result<()> {
    // omit validation for sf != 1.
    if scale_factor != 1 {
        return Ok(());
    }
    let query_dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../vortex-duckdb/duckdb/extension/tpch/dbgen/queries");

    let tmp_dir = format!(
        "{}/spiral-tpch",
        // $RUNNER_TEMP is defined by GitHub Actions.
        env::var("TMPDIR").or_else(|_| env::var("RUNNER_TEMP"))?
    );

    if PathBuf::from(&tmp_dir).exists() {
        fs::remove_dir_all(&tmp_dir)?;
    }
    fs::create_dir(&tmp_dir)?;
    let duckdb_ctx = ddb::DuckDBCtx::new_in_memory()?;
    duckdb_ctx.register_tables(
        url,
        Format::OnDiskVortex,
        &BenchmarkDataset::TpcH { scale_factor },
    )?;

    let mut query_files = fs::read_dir(query_dir)?
        .filter_map(Result::ok)
        .filter(|entry| entry.path().extension().is_some_and(|ext| ext == "sql"))
        .collect::<Vec<_>>();
    query_files.sort_by_key(|entry| entry.file_name());

    let mut is_matching_ref_result = true;

    for query_file in query_files
        .iter()
        .enumerate()
        .filter(|entry| {
            queries
                .as_ref()
                .is_none_or(|queries| queries.contains(&(entry.0 + 1)))
        })
        .map(|query_file| query_file.1)
    {
        let query_file_path = query_file.path();
        let query_name = query_file_path
            .file_stem()
            .and_then(|stem| stem.to_str())
            .ok_or_else(|| anyhow!("Invalid query filename"))?;

        let create_table = format!(
            "CREATE OR REPLACE TABLE {query_name}_result AS {};",
            fs::read_to_string(&query_file_path)?
        );

        let csv_actual = format!("{tmp_dir}/{query_name}.csv");
        let write_csv =
            format!("COPY {query_name}_result TO '{csv_actual}' (HEADER, DELIMITER '|');",);

        duckdb_ctx.execute_query(&create_table)?;
        duckdb_ctx.execute_query(&write_csv)?;

        let csv_expected = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join(format!("tpch/results/duckdb/{query_name}.csv"));
        let expected = fs::read_to_string(csv_expected)?;
        let actual = fs::read_to_string(csv_actual)?;

        if expected != actual {
            let diff = TextDiff::from_lines(&expected, &actual);

            for change in diff.iter_all_changes() {
                let sign = match change.tag() {
                    ChangeTag::Delete => "-",
                    ChangeTag::Insert => "+",
                    ChangeTag::Equal => " ",
                };
                print!("{}{}", sign, change);
            }

            eprintln!("query output does not match the reference for {query_name}");
            is_matching_ref_result = false;
        }
    }

    if !is_matching_ref_result {
        return Err(anyhow!("not all queries matched the reference"));
    }

    Ok(())
}

fn validate_args(engines: &[Engine], args: &Args) {
    if (args.all_metrics || args.export_spans || args.emit_plan || args.threads.is_some())
        && !engines.contains(&Engine::DataFusion)
    {
        vortex_panic!(
            "--all-metrics, --emit-plan, --threads, --export-spans are only valid if DataFusion is used"
        );
    }
}
