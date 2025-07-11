// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark driver that handles CLI logic and orchestrates benchmark execution

use std::path::PathBuf;

use anyhow::Result;
use indicatif::ProgressBar;
use log::warn;
use vortex_datafusion::metrics::VortexMetricsFinder;

use crate::benchmark_trait::Benchmark;
use crate::display::DisplayFormat;
use crate::engines::{EngineCtx, benchmark_datafusion_query, benchmark_duckdb_query};
use crate::measurements::{MemoryMeasurement, QueryMeasurement};
use crate::memory::MemoryTracker;
use crate::metrics::{MetricsSetExt, export_plan_spans};
use crate::query_bench::{filter_queries, print_results, setup_logging_and_tracing};
use crate::utils::{new_tokio_runtime, url_scheme_to_storage};
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
    pub track_memory: bool,
    pub force_memory_reclaim: bool,
}

/// Run a benchmark using the provided implementation and configuration
pub fn run_benchmark<B: Benchmark>(benchmark: B, config: DriverConfig) -> Result<()> {
    let _trace_guard = setup_logging_and_tracing(
        config.verbose,
        &format!("{}.trace.json", benchmark.dataset_name()),
    )?;

    // Generate data for each target (idempotent)
    for target in &config.targets {
        benchmark.generate_data(target)?;
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
    let mut all_memory_measurements = Vec::new();

    // Create a global memory tracker if memory tracking is enabled
    let global_memory_tracker = if config.track_memory {
        Some(MemoryTracker::new())
    } else {
        None
    };

    for target in config.targets.iter() {
        let tokio_runtime = new_tokio_runtime(config.threads);

        let mut engine_ctx = benchmark.setup_engine_context(
            target,
            config.disable_datafusion_cache,
            config.emit_plan,
        )?;

        tokio_runtime.block_on(benchmark.register_tables(&engine_ctx, target.format()))?;

        let (bench_measurements, memory_measurements) = execute_queries(
            &filtered_queries,
            config.iterations,
            &tokio_runtime,
            target.format(),
            &progress_bar,
            &mut engine_ctx,
            &benchmark,
            config.track_memory,
            config.force_memory_reclaim,
            global_memory_tracker.as_ref(),
        )?;

        tokio_runtime.block_on(export_metrics_if_requested(
            &engine_ctx,
            config.export_spans,
        ))?;

        if config.show_metrics {
            print_metrics(&engine_ctx);
        }

        query_measurements.extend(bench_measurements);
        all_memory_measurements.extend(memory_measurements);
    }

    // Print memory measurements if available
    if !all_memory_measurements.is_empty() && config.track_memory {
        println!("\n=== Memory Usage Summary ===");
        for memory_measurement in &all_memory_measurements {
            println!(
                "Query {}: Δ{}MB physical, Δ{}MB virtual {} | Reclaimed: {}MB physical, {}MB virtual | Peak: {}MB physical, {}MB virtual",
                memory_measurement.query_idx,
                memory_measurement.physical_memory_delta as f64 / 1024.0 / 1024.0,
                memory_measurement.virtual_memory_delta as f64 / 1024.0 / 1024.0,
                memory_measurement.target,
                memory_measurement.physical_memory_reclaimed.abs() as f64 / 1024.0 / 1024.0,
                memory_measurement.virtual_memory_reclaimed.abs() as f64 / 1024.0 / 1024.0,
                memory_measurement.peak_physical_memory as f64 / 1024.0 / 1024.0,
                memory_measurement.peak_virtual_memory as f64 / 1024.0 / 1024.0,
            );
        }
        println!("===========================\n");
    }

    print_results(
        &config.display_format,
        query_measurements,
        &config.targets,
        &config.output_path,
    )
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
    track_memory: bool,
    force_memory_reclaim: bool,
    global_memory_tracker: Option<&MemoryTracker>,
) -> Result<(Vec<QueryMeasurement>, Vec<MemoryMeasurement>)> {
    let mut query_measurements = Vec::new();
    let mut memory_measurements = Vec::new();
    let expected_row_counts = benchmark.expected_row_counts();

    for &(query_idx, ref query_string) in queries.iter() {
        // Get baseline memory before query if tracking is enabled
        let baseline_memory = if track_memory {
            global_memory_tracker.and_then(|tracker| tracker.current_memory())
        } else {
            None
        };

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
                    benchmark_dataset: benchmark.dataset(),
                    storage: url_scheme_to_storage(benchmark.data_url())?,
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
                    benchmark_dataset: benchmark.dataset(),
                    storage: url_scheme_to_storage(benchmark.data_url())?,
                    runs,
                });
            }
        }

        // Collect memory measurement if tracking is enabled
        if let (Some(baseline), Some(tracker)) = (baseline_memory, global_memory_tracker) {
            let after_memory = tracker.current_memory().unwrap_or_else(|| crate::memory::MemoryStats::new(0, 0));
            let usage_diff = baseline.diff(&after_memory);

            // Force memory reclamation if requested
            let reclaim_diff = if force_memory_reclaim {
                crate::memory::force_memory_reclaim();
                let after_reclaim = tracker.current_memory().unwrap_or_else(|| crate::memory::MemoryStats::new(0, 0));
                after_memory.diff(&after_reclaim)
            } else {
                crate::memory::MemoryStatsDiff {
                    physical_memory_delta: 0,
                    virtual_memory_delta: 0,
                }
            };

            // Get peak memory from global tracker
            let peak_memory = tracker.peak_memory();

            memory_measurements.push(MemoryMeasurement {
                query_idx,
                target: match engine_ctx {
                    EngineCtx::DataFusion(_) => Target::new(Engine::DataFusion, format),
                    EngineCtx::DuckDB(_) => Target::new(Engine::DuckDB, format),
                },
                benchmark_dataset: benchmark.dataset(),
                storage: url_scheme_to_storage(benchmark.data_url())?,
                physical_memory_delta: usage_diff.physical_memory_delta,
                virtual_memory_delta: usage_diff.virtual_memory_delta,
                physical_memory_reclaimed: reclaim_diff.physical_memory_delta,
                virtual_memory_reclaimed: reclaim_diff.virtual_memory_delta,
                peak_physical_memory: peak_memory.physical_memory,
                peak_virtual_memory: peak_memory.virtual_memory,
            });
        }

        progress_bar.inc(1);
    }

    Ok((query_measurements, memory_measurements))
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
