// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generic benchmark runner infrastructure to reduce boilerplate across engine-specific benchmarks.

use std::fs::File;
use std::future::Future;
use std::io::Write;
use std::path::PathBuf;
use std::pin::Pin;
use std::time::Duration;
use std::time::Instant;

use indicatif::ProgressBar;
use vortex::error::vortex_panic;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Engine;
use crate::Format;
use crate::Target;
use crate::display::DisplayFormat;
use crate::display::print_measurements_json;
use crate::display::render_table;
use crate::measurements::MemoryMeasurement;
use crate::measurements::QueryMeasurement;
use crate::memory::BenchmarkMemoryTracker;
use crate::url_scheme_to_storage;

/// Results from a benchmark run.
pub struct BenchmarkResults {
    pub query_measurements: Vec<QueryMeasurement>,
    pub memory_measurements: Vec<MemoryMeasurement>,
}

/// A benchmark runner that handles common scaffolding for SQL-oriented benchmarks:
/// - Progress bar management
/// - Memory tracking
/// - Measurement collection
/// - Row count validation
/// - Result export
///
/// Engine-specific code handles:
/// - Context creation
/// - Table registration
/// - Query execution
pub struct SqlBenchmarkRunner {
    engine: Engine,
    benchmark_dataset: BenchmarkDataset,
    storage: String,
    expected_row_counts: Option<Vec<usize>>,
    formats: Vec<Format>,
    memory_tracker: Option<BenchmarkMemoryTracker>,
    hide_progress_bar: bool,
    query_measurements: Vec<QueryMeasurement>,
    memory_measurements: Vec<MemoryMeasurement>,
}

impl SqlBenchmarkRunner {
    /// Create a new benchmark runner.
    pub fn new<B: Benchmark + ?Sized>(
        benchmark: &B,
        engine: Engine,
        formats: Vec<Format>,
        track_memory: bool,
        hide_progress_bar: bool,
    ) -> anyhow::Result<Self> {
        let storage = url_scheme_to_storage(benchmark.data_url())?;

        let memory_tracker = track_memory.then(BenchmarkMemoryTracker::new);

        Ok(Self {
            engine,
            benchmark_dataset: benchmark.dataset(),
            storage,
            expected_row_counts: benchmark.expected_row_counts().map(|s| s.to_vec()),
            formats,
            memory_tracker,
            hide_progress_bar,
            query_measurements: Vec::new(),
            memory_measurements: Vec::new(),
        })
    }

    /// Get the formats to run benchmarks for.
    pub fn formats(&self) -> &[Format] {
        &self.formats
    }

    /// Call before running a query to start memory tracking.
    fn start_query(&mut self) {
        if let Some(tracker) = self.memory_tracker.as_mut() {
            tracker.start_query();
        }
    }

    /// Run a synchronous query benchmark.
    ///
    /// Executes the query function `iterations` times, collecting timing information.
    /// The function should return `(row_count, optional_timing, result)` where:
    /// - `row_count` is used for validation
    /// - `optional_timing` can be `Some(Duration)` if the callback wants to report its own timing
    ///   (e.g., DuckDB's internal timing), or `None` to use external wall-clock measurement
    ///
    /// This handles:
    /// - Memory tracking (start/end)
    /// - Timing each iteration
    /// - Recording measurements
    /// - Row count validation
    /// - Progress bar updates
    fn run_query<F>(&mut self, query_idx: usize, format: Format, iterations: usize, mut f: F)
    where
        F: FnMut() -> (usize, Option<Duration>),
    {
        self.start_query();

        let mut runs = Vec::with_capacity(iterations);
        let mut result = None;

        for _ in 0..iterations {
            let start = Instant::now();
            let (row_count, timing) = f();
            let elapsed = timing.unwrap_or_else(|| start.elapsed());
            runs.push(elapsed);

            if result.is_none() {
                result = Some(row_count);
            }
        }

        let row_count = result.expect("iterations must be > 0");
        self.record_query(query_idx, format, runs, row_count);
    }

    /// Record the results of running a query.
    ///
    /// This will:
    /// - Store the query measurement
    /// - Validate row count if expected counts are available
    /// - Record memory measurement if tracking is enabled
    /// - Increment the progress bar
    fn record_query(
        &mut self,
        query_idx: usize,
        format: Format,
        runs: Vec<Duration>,
        row_count: usize,
    ) {
        let target = Target::new(self.engine, format);

        self.query_measurements.push(QueryMeasurement {
            query_idx,
            target,
            benchmark_dataset: self.benchmark_dataset.clone(),
            storage: self.storage.clone(),
            runs,
        });

        // Validate row count if expected counts are provided
        if let Some(expected_counts) = &self.expected_row_counts
            && query_idx < expected_counts.len()
        {
            assert_eq!(
                row_count,
                expected_counts[query_idx],
                "Row count mismatch for query {query_idx} - {engine}:{format}",
                engine = self.engine,
            );
        }

        // Record memory measurement if tracking is enabled
        if let Some(tracker) = self.memory_tracker.as_ref()
            && let Some(memory_result) = tracker.end_query()
        {
            self.memory_measurements.push(MemoryMeasurement::new(
                query_idx,
                target,
                self.benchmark_dataset.clone(),
                self.storage.clone(),
                memory_result,
            ));
        }
    }

    /// Export results to the specified output (file or stdout).
    pub fn export(
        self,
        output_path: Option<&PathBuf>,
        display_format: &DisplayFormat,
    ) -> anyhow::Result<()> {
        match output_path {
            Some(path) => {
                let f = File::create(path)?;
                export_results(
                    self.query_measurements,
                    self.memory_measurements,
                    display_format,
                    self.engine,
                    &self.formats,
                    f,
                )
            }
            None => export_results(
                self.query_measurements,
                self.memory_measurements,
                display_format,
                self.engine,
                &self.formats,
                std::io::stdout().lock(),
            ),
        }
    }

    /// Export results to a custom writer.
    pub fn export_to<W: Write>(
        self,
        display_format: &DisplayFormat,
        output: W,
    ) -> anyhow::Result<()> {
        export_results(
            self.query_measurements,
            self.memory_measurements,
            display_format,
            self.engine,
            &self.formats,
            output,
        )
    }

    /// Get the collected results without exporting.
    pub fn into_results(self) -> BenchmarkResults {
        BenchmarkResults {
            query_measurements: self.query_measurements,
            memory_measurements: self.memory_measurements,
        }
    }

    /// Run all queries for all formats synchronously.
    ///
    /// For each format:
    /// 1. Calls `setup` to create a context for that format
    /// 2. Iterates over all queries, calling `execute` for each
    ///
    /// The `execute` callback receives the context, query index, and query string,
    /// and should return `(row_count, optional_timing)` where `optional_timing` can be
    /// `Some(Duration)` if the callback wants to report its own timing.
    pub fn run_all<Ctx, S, E>(
        &mut self,
        queries: &[(usize, String)],
        iterations: usize,
        mut setup: S,
        mut execute: E,
    ) -> anyhow::Result<()>
    where
        S: FnMut(Format) -> anyhow::Result<Ctx>,
        E: FnMut(&mut Ctx, &str) -> anyhow::Result<(usize, Option<Duration>)>,
    {
        let bar_length = queries.len() * self.formats.len();
        let progress_bar = if self.hide_progress_bar || bar_length == 0 {
            ProgressBar::hidden()
        } else {
            ProgressBar::new(bar_length as u64)
        };

        for format in self.formats.clone() {
            let mut ctx = setup(format)?;

            for (query_idx, query) in queries.iter() {
                let query_idx = *query_idx;
                tracing::debug!(%format, query_idx, "Running query");
                self.run_query(query_idx, format, iterations, || {
                    let (row_count, timing) =
                        execute(&mut ctx, query.as_str()).unwrap_or_else(|err| {
                            vortex_panic!("query {query_idx} failed: {err}");
                        });
                    (row_count, timing)
                });

                progress_bar.inc(1);
            }
        }

        progress_bar.finish();

        Ok(())
    }

    /// Run all queries for all formats asynchronously.
    ///
    /// For each format:
    /// 1. Calls `setup` to create a context for that format
    /// 2. Iterates over all queries, calling `execute` for each
    ///
    /// The `execute` callback receives the context, query index, and query string,
    /// and should return `(row_count, optional_timing, result)` where `optional_timing` can be
    /// `Some(Duration)` if the callback wants to report its own timing.
    /// Use `Box::pin(async move { ... })` in the closure.
    pub async fn run_all_async<Ctx, S, SFut, E, T>(
        &mut self,
        queries: &[(usize, String)],
        iterations: usize,
        setup: S,
        mut execute: E,
    ) -> anyhow::Result<()>
    where
        S: Fn(Format) -> SFut,
        SFut: Future<Output = anyhow::Result<Ctx>>,
        E: for<'c> FnMut(
            &'c Ctx,
            &'c str,
        ) -> Pin<
            Box<dyn Future<Output = anyhow::Result<(usize, Option<Duration>, T)>> + 'c>,
        >,
    {
        let bar_length = queries.len() * self.formats.len();
        let progress_bar = if self.hide_progress_bar || bar_length == 0 {
            ProgressBar::hidden()
        } else {
            ProgressBar::new(bar_length as u64)
        };

        for format in self.formats.clone() {
            let ctx = setup(format).await?;

            for (query_idx, query) in queries.iter() {
                let query_idx = *query_idx;

                self.start_query();

                let mut runs = Vec::with_capacity(iterations);
                let mut result = None;

                tracing::debug!(%format, query_idx, "Running query");

                for _ in 0..iterations {
                    let start = Instant::now();
                    let (row_count, timing, iter_result) =
                        execute(&ctx, query.as_str()).await.unwrap_or_else(|err| {
                            vortex_panic!("query {query_idx} failed: {err}");
                        });
                    let elapsed = timing.unwrap_or_else(|| start.elapsed());
                    runs.push(elapsed);

                    if result.is_none() {
                        result = Some((row_count, iter_result));
                    }
                }

                let (row_count, _) = result.expect("iterations must be > 0");
                self.record_query(query_idx, format, runs, row_count);

                progress_bar.inc(1);
            }
        }

        progress_bar.finish();

        Ok(())
    }
}

pub fn export_results<W: Write>(
    queries: Vec<QueryMeasurement>,
    memory: Vec<MemoryMeasurement>,
    display_format: &DisplayFormat,
    engine: Engine,
    formats: &[Format],
    mut output: W,
) -> anyhow::Result<()> {
    let targets = formats
        .iter()
        .map(|f| Target::new(engine, *f))
        .collect::<Vec<_>>();

    if !memory.is_empty() {
        match display_format {
            DisplayFormat::Table => render_table(&mut output, memory, &targets)?,
            DisplayFormat::GhJson => print_measurements_json(&mut output, memory)?,
        };
    }

    match display_format {
        DisplayFormat::Table => render_table(&mut output, queries, &targets)?,
        DisplayFormat::GhJson => print_measurements_json(&mut output, queries)?,
    };

    Ok(())
}

/// Filter queries based on include/exclude lists
pub fn filter_queries(
    all_queries: Vec<(usize, String)>,
    include_queries: Option<&Vec<usize>>,
    exclude_queries: Option<&Vec<usize>>,
) -> Vec<(usize, String)> {
    all_queries
        .into_iter()
        .filter(|(query_idx, _)| {
            // Include query if:
            // 1. No specific queries were requested OR this query is in the requested list
            // 2. AND this query is not in the excluded list
            include_queries
                .as_ref()
                .is_none_or(|included| included.contains(query_idx))
                && exclude_queries
                    .as_ref()
                    .is_none_or(|excluded| !excluded.contains(query_idx))
        })
        .collect()
}
