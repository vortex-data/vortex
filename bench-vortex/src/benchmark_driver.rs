// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark driver that handles CLI logic and orchestrates benchmark execution

use std::path::PathBuf;

use anyhow::Result;
use indicatif::ProgressBar;
use itertools::Itertools;
use log::warn;
use url::Url;
use vortex_datafusion::metrics::VortexMetricsFinder;

use crate::benchmark_trait::Benchmark;
use crate::display::DisplayFormat;
use crate::engines::{EngineCtx, benchmark_datafusion_query, benchmark_duckdb_query};
use crate::measurements::QueryMeasurement;
use crate::metrics::{MetricsSetExt, export_plan_spans};
use crate::unified::{filter_queries, print_results, setup_logging_and_tracing};
use crate::utils::constants::STORAGE_NVME;
use crate::utils::new_tokio_runtime;
use crate::{Engine, Format, Target, df, vortex_panic};

/// Configuration for the benchmark driver
pub struct DriverConfig {
    pub targets: Vec<Target>,
    pub iterations: usize,
    pub threads: Option<usize>,
    pub verbose: bool,
    pub display_format: DisplayFormat,
    pub disable_datafusion_cache: bool,
    pub queries: Option<Vec<usize>>,
    pub exclude_queries: Option<Vec<usize>>,
    pub output_path: Option<PathBuf>,
    pub emit_plan: bool,
    pub export_spans: bool,
    pub show_metrics: bool,
    pub hide_progress_bar: bool,
}

/// Run a benchmark using the provided implementation and configuration
pub fn run_benchmark<B: Benchmark>(
    benchmark: B,
    config: DriverConfig,
    trace_file_name: &str,
    data_url: Url,
) -> Result<()> {
    let _trace_guard = setup_logging_and_tracing(config.verbose, trace_file_name)?;

    // Validate arguments
    validate_args(&config)?;

    // Generate data for each target (idempotent)
    for target in &config.targets {
        benchmark.generate_data(&data_url, target)?;
    }

    let filtered_queries = filter_queries(
        benchmark.queries()?,
        config.queries.as_ref(),
        config.exclude_queries.as_ref(),
    );

    let progress_bar = if config.hide_progress_bar {
        ProgressBar::hidden()
    } else {
        ProgressBar::new((filtered_queries.len() * config.targets.len()) as u64)
    };

    let mut query_measurements = Vec::new();

    for target in config.targets.iter() {
        let tokio_runtime = new_tokio_runtime(config.threads);

        let mut engine_ctx = setup_engine_context(
            target,
            &data_url,
            config.disable_datafusion_cache,
            config.emit_plan,
        )?;

        // Register tables
        tokio_runtime.block_on(benchmark.register_tables(
            &engine_ctx,
            &data_url,
            target.format(),
        ))?;

        // Execute queries
        let bench_measurements = execute_queries(
            &filtered_queries,
            config.iterations,
            &tokio_runtime,
            target.format(),
            &progress_bar,
            &mut engine_ctx,
            &benchmark,
        );

        // Export metrics and spans
        tokio_runtime.block_on(export_metrics_if_requested(
            &engine_ctx,
            config.export_spans,
        ))?;

        // Print metrics if requested
        if config.show_metrics {
            print_metrics(&engine_ctx);
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

fn validate_args(config: &DriverConfig) -> Result<()> {
    let engines = config
        .targets
        .iter()
        .map(|t| t.engine())
        .unique()
        .collect_vec();

    if (config.emit_plan || config.export_spans || config.show_metrics || config.threads.is_some())
        && !engines.contains(&Engine::DataFusion)
    {
        vortex_panic!(
            "--emit-plan, --export-spans, --show-metrics, --threads are only valid if DataFusion is used"
        );
    }
    Ok(())
}

fn setup_engine_context(
    target: &Target,
    data_url: &Url,
    disable_datafusion_cache: bool,
    emit_plan: bool,
) -> Result<EngineCtx> {
    let engine = target.engine();
    let format = target.format();

    match engine {
        Engine::DataFusion => {
            let session_ctx = df::get_session_context(disable_datafusion_cache);
            df::make_object_store(&session_ctx, data_url)?;
            Ok(EngineCtx::new_with_datafusion(session_ctx, emit_plan))
        }
        Engine::DuckDB => {
            // Create a generic dataset for DuckDB context creation
            // This will be properly configured when tables are registered
            let dataset = crate::BenchmarkDataset::ClickBench {
                single_file: false,
                flavor: crate::clickbench::Flavor::Partitioned,
            };
            Ok(EngineCtx::new_with_duckdb(dataset, format)?)
        }
        _ => unreachable!("engine not supported"),
    }
}

#[allow(clippy::too_many_arguments)]
fn execute_queries<B: Benchmark>(
    queries: &[(usize, String)],
    iterations: usize,
    runtime: &tokio::runtime::Runtime,
    format: Format,
    progress_bar: &ProgressBar,
    engine_ctx: &mut EngineCtx,
    benchmark: &B,
) -> Vec<QueryMeasurement> {
    let mut query_measurements = Vec::new();
    let expected_row_counts = benchmark.get_expected_row_counts();

    for &(query_idx, ref query_string) in queries.iter() {
        match engine_ctx {
            EngineCtx::DataFusion(ctx) => {
                let (runs, (row_count, execution_plan)) = runtime.block_on(async {
                    benchmark_datafusion_query(iterations, || async {
                        let (batches, plan) = df::execute_query(&ctx.session, query_string)
                            .await
                            .unwrap_or_else(|err| {
                                vortex_panic!("query: {query_idx} failed with: {err}")
                            });
                        let row_count: usize = batches.iter().map(|batch| batch.num_rows()).sum();
                        (row_count, plan)
                    })
                    .await
                });

                // Validate row count if expected counts are provided
                if let Some(expected_counts) = expected_row_counts {
                    if query_idx < expected_counts.len() {
                        assert_eq!(
                            row_count, expected_counts[query_idx],
                            "Row count mismatch for query {query_idx} - datafusion:{format}",
                        );
                    }
                }

                ctx.execution_plans
                    .push((query_idx, execution_plan.clone()));

                if ctx.emit_plan {
                    df::write_execution_plan(
                        query_idx,
                        format,
                        benchmark.dataset_name(),
                        execution_plan.as_ref(),
                    );
                }

                ctx.metrics.push((
                    query_idx,
                    format,
                    VortexMetricsFinder::find_all(execution_plan.as_ref()),
                ));

                query_measurements.push(QueryMeasurement {
                    query_idx,
                    target: Target::new(Engine::DataFusion, format),
                    benchmark_dataset: benchmark.get_dataset(),
                    storage: STORAGE_NVME.to_owned(),
                    runs,
                });
            }
            EngineCtx::DuckDB(ctx) => {
                let (runs, row_count) =
                    benchmark_duckdb_query(query_idx, query_string, iterations, ctx);

                // Validate row count if expected counts are provided
                if let Some(expected_counts) = expected_row_counts {
                    if query_idx < expected_counts.len() {
                        assert_eq!(
                            row_count, expected_counts[query_idx],
                            "Row count mismatch for query {query_idx} - duckdb:{format}",
                        );
                    }
                }

                query_measurements.push(QueryMeasurement {
                    query_idx,
                    target: Target::new(Engine::DuckDB, format),
                    benchmark_dataset: benchmark.get_dataset(),
                    storage: STORAGE_NVME.to_owned(),
                    runs,
                });
            }
        }

        progress_bar.inc(1);
    }

    query_measurements
}

async fn export_metrics_if_requested(engine_ctx: &EngineCtx, export_spans: bool) -> Result<()> {
    if let EngineCtx::DataFusion(ctx) = engine_ctx {
        if export_spans {
            if let Err(err) = export_plan_spans(Format::OnDiskVortex, &ctx.execution_plans).await {
                warn!("failed to export spans {err}");
            }
        }
    }
    Ok(())
}

fn print_metrics(engine_ctx: &EngineCtx) {
    if let EngineCtx::DataFusion(ctx) = engine_ctx {
        for (query_idx, file_format, metric_sets) in &ctx.metrics {
            eprintln!("metrics for query={query_idx}, {file_format}:");
            for (scan_idx, metrics_set) in metric_sets.iter().enumerate() {
                eprintln!("scan[{scan_idx}]:");
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
}
