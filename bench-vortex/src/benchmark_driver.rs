// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Benchmark driver that handles CLI logic and orchestrates benchmark execution

use std::fmt;
use std::path::PathBuf;

use anyhow::Result;
use indicatif::ProgressBar;
use log::warn;
use vortex::error::VortexExpect;
use vortex_datafusion::metrics::VortexMetricsFinder;

use crate::benchmark_trait::Benchmark;
use crate::display::DisplayFormat;
use crate::engines::{EngineCtx, benchmark_datafusion_query};
use crate::measurements::{MemoryMeasurement, QueryMeasurement};
use crate::memory::BenchmarkMemoryTracker;
use crate::metrics::{MetricsSetExt, export_plan_spans};
use crate::query_bench::{filter_queries, print_memory_usage, print_results};
use crate::utils::{new_tokio_runtime, url_scheme_to_storage};
use crate::{Engine, Format, Target, df, vortex_panic};

/// Mode for EXPLAIN queries
#[derive(Debug, Clone, Copy)]
enum ExplainMode {
    /// EXPLAIN - show query plan without execution
    Plan,
    /// EXPLAIN ANALYZE - execute query and show plan with metrics
    Analyze,
}

impl fmt::Display for ExplainMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ExplainMode::Plan => write!(f, "EXPLAIN"),
            ExplainMode::Analyze => write!(f, "EXPLAIN ANALYZE"),
        }
    }
}

/// Configuration for the benchmark driver
pub struct DriverConfig {
    pub targets: Vec<Target>,
    pub iterations: usize,
    pub threads: Option<usize>,
    pub display_format: DisplayFormat,
    pub disable_datafusion_cache: bool,
    pub delete_duckdb_database: bool,
    pub queries: Option<Vec<usize>>,
    pub exclude_queries: Option<Vec<usize>>,
    pub output_path: Option<PathBuf>,
    pub emit_plan: bool,
    pub export_spans: bool,
    pub show_metrics: bool,
    pub hide_progress_bar: bool,
    pub track_memory: bool,
    pub skip_generate: bool,
    pub explain: bool,
    pub explain_analyze: bool,
}

/// Run a benchmark using the provided implementation and configuration
pub fn run_benchmark<B: Benchmark>(benchmark: B, config: DriverConfig) -> Result<()> {
    // If explain-analyze mode is enabled, run explain analyze
    if config.explain_analyze {
        return run_explain_query(benchmark, config, ExplainMode::Analyze);
    }

    // If explain mode is enabled, run explain (without execution)
    if config.explain {
        return run_explain_query(benchmark, config, ExplainMode::Plan);
    }

    // Generate data for each target (idempotent)
    if !config.skip_generate {
        for target in &config.targets {
            benchmark.generate_data(target)?;
        }
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
    let mut global_memory_tracker = config.track_memory.then(BenchmarkMemoryTracker::new);

    for target in config.targets.iter() {
        let tokio_runtime = new_tokio_runtime(config.threads)?;

        let mut engine_ctx = benchmark.setup_engine_context(
            target,
            config.disable_datafusion_cache,
            config.emit_plan,
            config.delete_duckdb_database,
            config.threads,
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
            global_memory_tracker.as_mut(),
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
        print_memory_usage(
            all_memory_measurements,
            &config.display_format,
            &config.targets,
        )?;
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
    mut global_memory_tracker: Option<&mut BenchmarkMemoryTracker>,
) -> Result<(Vec<QueryMeasurement>, Vec<MemoryMeasurement>)> {
    let mut query_measurements = Vec::new();
    let mut memory_measurements = Vec::new();
    let expected_row_counts = benchmark.expected_row_counts();

    for &(query_idx, ref query_string) in queries.iter() {
        // Start memory tracking before query
        if let Some(tracker) = global_memory_tracker.as_mut() {
            tracker.start_query();
        }

        let row_count = match engine_ctx {
            EngineCtx::DataFusion(ctx) => {
                let (runs, (row_count, execution_plan)) = runtime.block_on(async {
                    benchmark_datafusion_query(iterations, || async {
                        let (batches, plan) =
                            ctx.execute_query(query_string).await.unwrap_or_else(|err| {
                                vortex_panic!("query: {query_idx} failed with: {err}")
                            });
                        let row_count: usize = batches.iter().map(|batch| batch.num_rows()).sum();
                        (row_count, plan)
                    })
                    .await
                });

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

                row_count
            }
            EngineCtx::DuckDB(ctx) => {
                let mut runs = Vec::with_capacity(iterations);
                let mut row_count = None;

                for _ in 0..iterations {
                    // Ensure we reopen the database to clear caches between runs.
                    ctx.reopen()?;

                    let (duration, current_row_count) =
                        ctx.execute_query(query_string).unwrap_or_else(|err| {
                            vortex_panic!("query: {query_idx} failed with: {err}")
                        });

                    runs.push(duration);
                    row_count.inspect(|rc| {
                        assert_eq!(
                            *rc, current_row_count,
                            "Row count mismatch between iterations for query {}: previous => {} rows vs. current => {} rows",
                            query_idx, *rc, current_row_count
                        )
                    });
                    row_count = Some(current_row_count);
                }

                query_measurements.push(QueryMeasurement {
                    query_idx,
                    target: Target::new(Engine::DuckDB, format),
                    benchmark_dataset: benchmark.dataset(),
                    storage: url_scheme_to_storage(benchmark.data_url())?,
                    runs,
                });

                row_count.vortex_expect("cannot have zero runs")
            }
        };

        // Validate row count if expected counts are provided
        if let Some(expected_counts) = expected_row_counts
            && query_idx < expected_counts.len()
        {
            assert_eq!(
                row_count,
                expected_counts[query_idx],
                "Row count mismatch for query {query_idx} - {}:{format}",
                engine_ctx.to_engine()
            );
        }

        // End memory tracking after query and collect measurements
        if let Some(tracker) = global_memory_tracker.as_ref()
            && let Some(memory_result) = tracker.end_query()
        {
            memory_measurements.push(MemoryMeasurement {
                query_idx,
                target: Target::new(engine_ctx.to_engine(), format),
                benchmark_dataset: benchmark.dataset(),
                storage: url_scheme_to_storage(benchmark.data_url())?,
                physical_memory_delta: memory_result.physical_memory_delta,
                virtual_memory_delta: memory_result.virtual_memory_delta,
                peak_physical_memory: memory_result.peak_physical_memory,
                peak_virtual_memory: memory_result.peak_virtual_memory,
            });
        }

        progress_bar.inc(1);
    }

    Ok((query_measurements, memory_measurements))
}

async fn export_metrics_if_requested(engine_ctx: &EngineCtx, export_spans: bool) -> Result<()> {
    if let EngineCtx::DataFusion(ctx) = engine_ctx
        && export_spans
        && let Err(err) = export_plan_spans(Format::OnDiskVortex, &ctx.execution_plans).await
    {
        warn!("failed to export spans {err}");
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

/// Run EXPLAIN or EXPLAIN ANALYZE for each query
fn run_explain_query<B: Benchmark>(
    benchmark: B,
    config: DriverConfig,
    mode: ExplainMode,
) -> Result<()> {
    // Generate data for each target (idempotent)
    if !config.skip_generate {
        for target in &config.targets {
            benchmark.generate_data(target)?;
        }
    }

    let filtered_queries = filter_queries(
        benchmark.queries()?,
        config.queries.as_ref(),
        config.exclude_queries.as_ref(),
    );

    for target in config.targets.iter() {
        println!("\n{}", "=".repeat(80));
        println!("Target: {} - {}", target.engine(), target.format());
        println!("{}\n", "=".repeat(80));

        let tokio_runtime = new_tokio_runtime(config.threads)?;

        let engine_ctx = benchmark.setup_engine_context(
            target,
            config.disable_datafusion_cache,
            config.emit_plan,
            config.delete_duckdb_database,
            config.threads,
        )?;

        tokio_runtime.block_on(benchmark.register_tables(&engine_ctx, target.format()))?;

        for &(query_idx, ref query_string) in filtered_queries.iter() {
            println!("\n{}", "-".repeat(80));
            println!("Query {}", query_idx);
            println!("{}", "-".repeat(80));
            println!("SQL: {}\n", query_string);

            match &engine_ctx {
                EngineCtx::DataFusion(ctx) => {
                    let explain_query = format!("{} {}", mode, query_string);
                    match tokio_runtime.block_on(async {
                        let plan = ctx.session.sql(&explain_query).await?;
                        let batches = plan.collect().await?;
                        Ok::<_, anyhow::Error>(batches)
                    }) {
                        Ok(batches) => {
                            for batch in batches {
                                for row_idx in 0..batch.num_rows() {
                                    for col_idx in 0..batch.num_columns() {
                                        let array = batch.column(col_idx);
                                        let string_array =
                                            arrow_array::cast::as_string_array(array);
                                        if let Some(value) =
                                            string_array.iter().nth(row_idx).flatten()
                                        {
                                            println!("{}", value);
                                        }
                                    }
                                }
                            }
                        }
                        Err(err) => {
                            eprintln!("Error running {} for query {}: {}", mode, query_idx, err);
                        }
                    }
                }
                EngineCtx::DuckDB(ctx) => {
                    let explain_query = format!("{} {}", mode, query_string);
                    match ctx.connection.query(&explain_query) {
                        Ok(result) => {
                            // DuckDB EXPLAIN returns a result set with data chunks
                            for chunk in result {
                                match String::try_from(&chunk) {
                                    Ok(output) => println!("{}", output),
                                    Err(err) => {
                                        eprintln!("Error converting chunk to string: {}", err)
                                    }
                                }
                            }
                        }
                        Err(err) => {
                            eprintln!("Error running {} for query {}: {}", mode, query_idx, err);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
