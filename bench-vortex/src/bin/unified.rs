// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::path::{Path, PathBuf};
use std::env;

use anyhow::anyhow;

use bench_vortex::clickbench::{Flavor, clickbench_queries};
use bench_vortex::display::DisplayFormat;
use bench_vortex::engines::{EngineCtx, benchmark_datafusion_query, benchmark_duckdb_query, ddb};
use bench_vortex::measurements::QueryMeasurement;
use bench_vortex::metrics::{MetricsSetExt, export_plan_spans};
use bench_vortex::public_bi::{PBIDataset, PBI_DATASETS, FileType};
use bench_vortex::tpch::duckdb::{DuckdbTpcOptions, TpcDataset, generate_tpc};
use bench_vortex::tpch::{EXPECTED_ROW_COUNTS_SF1, EXPECTED_ROW_COUNTS_SF10, load_datasets, run_tpch_query, tpch_queries};
use bench_vortex::unified::{BenchmarkConfig, setup_logging_and_tracing, print_results};
use bench_vortex::utils::constants::{CLICKBENCH_DATASET, STORAGE_NVME};
use bench_vortex::utils::new_tokio_runtime;
use bench_vortex::{BenchmarkDataset, Engine, Format, IdempotentPath, Target, df, vortex_panic};
use clap::{Parser, Subcommand, ValueEnum, value_parser};
use datafusion::physical_plan::metrics::{Label, MetricsSet};
use indicatif::ProgressBar;
use itertools::Itertools;
use log::{info, warn};
use similar::{ChangeTag, TextDiff};
use std::fs;
use tracing::debug;
use tracing_futures::Instrument;
use url::Url;
use vortex::error::VortexExpect;
use vortex_datafusion::metrics::VortexMetricsFinder;

#[derive(Parser, Debug)]
#[command(version, about = "Unified Vortex benchmark runner", long_about = None)]
struct Args {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// Run ClickBench queries
    #[command(name = "clickbench")]
    ClickBench(ClickBenchArgs),
    
    /// Run TPC-H queries
    #[command(name = "tpch")]
    TpcH(TpcHArgs),
    
    /// Run TPC-DS queries  
    #[command(name = "tpcds")]
    TpcDS(TpcDSArgs),
    
    /// Run Public BI queries
    #[command(name = "public-bi")]
    PublicBi(PublicBiArgs),
}

/// Common arguments shared across benchmarks
#[derive(Parser, Debug)]
struct CommonArgs {
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
    disable_datafusion_cache: bool,
    
    #[arg(short, long, value_delimiter = ',')]
    queries: Option<Vec<usize>>,
    
    #[arg(short)]
    output_path: Option<PathBuf>,
}

#[derive(Parser, Debug)]
struct ClickBenchArgs {
    #[command(flatten)]
    common: CommonArgs,
    
    #[arg(long, default_value_t = false)]
    emit_plan: bool,
    
    #[arg(long)]
    queries_file: Option<PathBuf>,
    
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

#[derive(Parser, Debug)]
struct TpcHArgs {
    #[command(flatten)]
    common: CommonArgs,
    
    #[arg(short, long, value_delimiter = ',')]
    exclude_queries: Option<Vec<usize>>,
    
    #[arg(long, default_value_t = 1, value_parser=validate_scale_factor)]
    scale_factor: u32,
    
    #[arg(long, default_value_t, value_enum)]
    data_generator: DataGenerator,
    
    #[arg(long)]
    all_metrics: bool,
    
    #[arg(long)]
    export_spans: bool,
    
    #[arg(long, default_value_t = false)]
    emit_plan: bool,
    
    #[arg(long)]
    use_remote_data_dir: Option<String>,
}

#[derive(Parser, Debug)]
struct TpcDSArgs {
    #[command(flatten)]
    common: CommonArgs,
    
    #[arg(short, long, value_delimiter = ',')]
    exclude_queries: Option<Vec<usize>>,
    
    #[arg(long)]
    export_spans: bool,
    
    #[arg(long, default_value_t = false)]
    emit_plan: bool,
}

#[derive(Parser, Debug)]
struct PublicBiArgs {
    #[command(flatten)]
    common: CommonArgs,
    
    #[arg(long)]
    display_metrics: bool,
    
    #[arg(short, long, value_delimiter = ',')]
    dataset: PBIDataset,
}

#[derive(ValueEnum, Default, Clone, Debug, PartialEq, Eq)]
pub enum DataGenerator {
    #[default]
    Dbgen,
    Duckdb,
}

fn validate_scale_factor(val: &str) -> Result<u32, String> {
    match val.parse::<u32>() {
        Ok(n) if [1, 10, 100, 1000].contains(&n) => Ok(n),
        _ => Err(String::from(
            "Value must be a scale factor of 1, 10, 100 or 1000",
        )),
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();
    
    match args.command {
        Commands::ClickBench(clickbench_args) => {
            run_clickbench(clickbench_args)
        }
        Commands::TpcH(tpch_args) => {
            run_tpch(tpch_args) 
        }
        Commands::TpcDS(tpcds_args) => {
            run_tpcds(tpcds_args)
        }
        Commands::PublicBi(public_bi_args) => {
            run_public_bi(public_bi_args)
        }
    }
}

fn run_clickbench(args: ClickBenchArgs) -> anyhow::Result<()> {
    let config = BenchmarkConfig {
        targets: args.common.targets.clone(),
        iterations: args.common.iterations,
        threads: args.common.threads,
        verbose: args.common.verbose,
        display_format: args.common.display_format.clone(),
        disable_datafusion_cache: args.common.disable_datafusion_cache,
        queries: args.common.queries.clone(),
        output_path: args.common.output_path.clone(),
    };

    let _trace_guard = setup_logging_and_tracing(config.verbose, "clickbench.trace.json")?;

    let engines = config.targets.iter().map(|t| t.engine()).unique().collect_vec();
    validate_clickbench_args(&engines, &args);

    let queries_filepath = args.queries_file.unwrap_or_else(|| {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("clickbench_queries.sql")
    });

    debug!(file = ?queries_filepath, "Reading queries from file");

    let queries = match &config.queries {
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
        ProgressBar::new((queries.len() * config.targets.len()) as u64)
    };

    let mut query_measurements = Vec::new();
    let dataset = BenchmarkDataset::ClickBench {
        single_file: args.single_file,
        flavor: args.flavor,
    };

    for target in config.targets.iter() {
        let engine = target.engine();
        let format = target.format();

        let mut engine_ctx = match engine {
            Engine::DataFusion => {
                let session_ctx = df::get_session_context(config.disable_datafusion_cache);
                df::make_object_store(&session_ctx, &base_url)?;
                EngineCtx::new_with_datafusion(session_ctx, args.emit_plan)
            }
            Engine::DuckDB => EngineCtx::new_with_duckdb(dataset.clone(), format)?,
            _ => unreachable!("engine not supported"),
        };

        let tokio_runtime = new_tokio_runtime(config.threads);
        tokio_runtime.block_on(init_clickbench_data_source(format, &base_url, &dataset, &engine_ctx))?;

        let bench_measurements = execute_clickbench_queries(
            &queries,
            config.iterations,
            &tokio_runtime,
            format,
            dataset.clone(),
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
                print_clickbench_metrics(&ctx.metrics);
            }
        }

        query_measurements.extend(bench_measurements);
    }

    print_results(
        &config.display_format,
        query_measurements,
        &config.targets,
        &config.output_path,
    )
}

fn run_tpch(args: TpcHArgs) -> anyhow::Result<()> {
    let config = BenchmarkConfig {
        targets: args.common.targets.clone(),
        iterations: args.common.iterations,
        threads: args.common.threads,
        verbose: args.common.verbose,
        display_format: args.common.display_format.clone(),
        disable_datafusion_cache: args.common.disable_datafusion_cache,
        queries: args.common.queries.clone(),
        output_path: args.common.output_path.clone(),
    };

    let _trace_guard = setup_logging_and_tracing(config.verbose, "tpch.trace.json")?;

    let engines = config.targets.iter().map(|t| t.engine()).collect_vec();
    validate_tpch_args(&engines, &args);

    let formats = config.targets.iter().map(|t| t.format()).collect_vec();
    let runtime = new_tokio_runtime(config.threads);

    let url = match args.use_remote_data_dir {
        None => {
            for format in formats {
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

    runtime.block_on(bench_tpch_main(
        config.queries,
        args.exclude_queries,
        config.iterations,
        config.targets,
        config.display_format,
        config.disable_datafusion_cache,
        args.scale_factor,
        url,
        args.all_metrics,
        args.export_spans,
        args.emit_plan,
        &config.output_path,
    ))
}

fn run_tpcds(args: TpcDSArgs) -> anyhow::Result<()> {
    let config = BenchmarkConfig {
        targets: args.common.targets.clone(),
        iterations: args.common.iterations,
        threads: args.common.threads,
        verbose: args.common.verbose,
        display_format: args.common.display_format.clone(),
        disable_datafusion_cache: args.common.disable_datafusion_cache,
        queries: args.common.queries.clone(),
        output_path: args.common.output_path.clone(),
    };

    let _trace_guard = setup_logging_and_tracing(config.verbose, "tpcds.trace.json")?;

    let formats = config.targets.iter().map(|t| t.format()).unique().collect_vec();

    for format in formats {
        let opts = DuckdbTpcOptions::new("tpcds".to_data_path(), TpcDataset::TpcDs, format);
        generate_tpc(opts).expect("gen tpch-ds");
    }

    let url = Url::parse(
        format!(
            "file:{}/{}/",
            "tpcds".to_data_path().to_str().vortex_expect("path must be utf8"),
            1 // scale factor 1
        )
        .as_ref(),
    )?;

    let runtime = new_tokio_runtime(None);

    runtime.block_on(bench_tpcds_main(
        config.queries,
        args.exclude_queries,
        config.iterations,
        config.targets,
        1, // scale factor 1 for now
        config.display_format,
        url,
        &config.output_path,
    ))
}

fn run_public_bi(args: PublicBiArgs) -> anyhow::Result<()> {
    let config = BenchmarkConfig {
        targets: args.common.targets.clone(),
        iterations: args.common.iterations,
        threads: args.common.threads,
        verbose: args.common.verbose,
        display_format: args.common.display_format.clone(),
        disable_datafusion_cache: args.common.disable_datafusion_cache,
        queries: args.common.queries.clone(),
        output_path: args.common.output_path.clone(),
    };

    let _trace_guard = setup_logging_and_tracing(config.verbose, "public_bi.trace.json")?;

    let runtime = new_tokio_runtime(config.threads);

    let pbi_dataset = PBI_DATASETS.get(args.dataset);
    let queries = match config.queries.clone() {
        None => pbi_dataset.queries()?,
        Some(queries) => pbi_dataset
            .queries()?
            .into_iter()
            .filter(|(q_idx, _)| queries.iter().contains(q_idx))
            .collect(),
    };

    let progress_bar = ProgressBar::new((queries.len() * config.targets.len()) as u64);
    let mut all_measurements = Vec::default();
    let mut metrics = Vec::new();

    let dataset = pbi_dataset.dataset()?;
    tracing::info!("preparing files");
    runtime.block_on(dataset.write_as_vortex());

    for target in &config.targets {
        let format = target.format();
        let session = df::get_session_context(config.disable_datafusion_cache);

        let file_type = match format {
            Format::Csv => FileType::Csv,
            Format::Parquet => FileType::Parquet,
            Format::OnDiskVortex => FileType::Vortex,
            other => vortex_panic!("Format {other} isn't supported on Public BI"),
        };

        runtime.block_on(dataset.register_tables(&session, file_type))?;

        for (query_idx, query) in queries.clone().into_iter() {
            let mut runs = Vec::with_capacity(config.iterations);
            let mut last_plan = None;
            
            for iteration in 0..config.iterations {
                let exec_duration = runtime.block_on(async {
                    let start = std::time::Instant::now();
                    let context = session.clone();
                    let query = query.clone();
                    last_plan = tokio::task::spawn(async move {
                        Some(
                            df::execute_query(&context, &query)
                                .instrument(tracing::info_span!("execute_query", query_idx, iteration))
                                .await
                                .unwrap_or_else(|e| {
                                    vortex_panic!("executing query {query_idx}: {e}")
                                })
                                .1,
                        )
                    })
                    .await
                    .vortex_expect("Failed to spawn query");

                    start.elapsed()
                });
                runs.push(exec_duration);
            }

            let plan = last_plan.vortex_expect("must have at least one iteration");

            metrics.push((
                query_idx,
                format,
                VortexMetricsFinder::find_all(plan.as_ref()),
            ));

            all_measurements.push(QueryMeasurement {
                query_idx,
                target: *target,
                benchmark_dataset: BenchmarkDataset::PublicBi {
                    name: pbi_dataset.name.clone(),
                },
                storage: STORAGE_NVME.to_owned(),
                runs,
            });

            progress_bar.inc(1);
        }
    }

    if args.display_metrics {
        for (query, format, metric_sets) in metrics {
            println!("\nmetrics for query={query}, {format}:");
            for (idx, metric_set) in metric_sets.into_iter().enumerate() {
                println!("scan[{idx}]:");
                for m in metric_set
                    .timestamps_removed()
                    .aggregate()
                    .sorted_for_display()
                    .iter()
                {
                    println!("{m}");
                }
            }
        }
    }

    print_results(
        &config.display_format,
        all_measurements,
        &config.targets,
        &config.output_path,
    )
}

// Helper functions extracted from the original binaries

fn validate_clickbench_args(engines: &[Engine], args: &ClickBenchArgs) {
    if (args.emit_plan || args.export_spans || args.show_metrics || args.common.threads.is_some())
        && !engines.contains(&Engine::DataFusion)
    {
        vortex_panic!(
            "--emit-plan, --export-spans, --show_metrics, --threads are only valid if DataFusion is used"
        );
    }
}

fn validate_tpch_args(engines: &[Engine], args: &TpcHArgs) {
    if (args.all_metrics || args.export_spans || args.emit_plan || args.common.threads.is_some())
        && !engines.contains(&Engine::DataFusion)
    {
        vortex_panic!(
            "--all-metrics, --emit-plan, --threads, --export-spans are only valid if DataFusion is used"
        );
    }
}

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
            if !remote_data_dir.ends_with("/") {
                log::warn!(
                    "Supply a --use-remote-data-dir argument which ends in a slash e.g. s3://vortex-bench-dev-eu/parquet/"
                );
            }
            log::info!(
                concat!(
                    "Assuming data already exists at this remote (e.g. S3, GCS) URL: {}.\\n",
                    "If it does not, you should kill this command, locally generate the files (by running without\\n",
                    "--use-remote-data-dir) and upload data/clickbench/ to some remote location.",
                ),
                remote_data_dir,
            );
            Ok(Url::parse(remote_data_dir)?)
        }
    }
}

async fn init_clickbench_data_source(
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

#[allow(clippy::too_many_arguments)]
fn execute_clickbench_queries(
    queries: &[(usize, String)],
    iterations: usize,
    tokio_runtime: &tokio::runtime::Runtime,
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
                let (runs, (execution_plan, row_count)) = tokio_runtime.block_on(async {
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
                    runs,
                });
            }
            EngineCtx::DuckDB(ctx) => {
                let (runs, row_count) =
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
                    runs,
                });
            }
        };

        progress_bar.inc(1);
    }

    query_measurements
}

fn print_clickbench_metrics(
    metrics: &Vec<(
        usize,
        Format,
        Vec<MetricsSet>,
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

// Placeholder async main functions for TPCH and TPCDS
// In a real implementation, these would contain the extracted logic from the original binaries

#[allow(clippy::too_many_arguments)]
async fn bench_tpch_main(
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
                        df::write_execution_plan(query_idx, format, dataset.name(), plan.as_ref());
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

    let mut writer: Box<dyn std::io::Write> = if let Some(output_path) = output_path {
        Box::new(fs::File::create(output_path)?)
    } else {
        let stdout = std::io::stdout();
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
            bench_vortex::display::render_table(&mut writer, measurements, &targets)?;
        }
        DisplayFormat::GhJson => {
            bench_vortex::display::print_measurements_json(&mut writer, measurements)?;
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

#[allow(clippy::too_many_arguments)]
async fn bench_tpcds_main(
    _queries: Option<Vec<usize>>,
    _exclude_queries: Option<Vec<usize>>,
    _iterations: usize,
    _targets: Vec<Target>,
    _scale_factor: u32,
    _display_format: DisplayFormat,
    _url: Url,
    _output_path: &Option<PathBuf>,
) -> anyhow::Result<()> {
    // TODO: Extract and implement the logic from the original tpcds.rs
    todo!("Implement TPCDS benchmark logic")
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