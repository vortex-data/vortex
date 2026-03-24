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
use crate::validation::validate_against_slt;

/// Controls whether queries are benchmarked or explained.
pub enum BenchmarkMode {
    /// Run each query `iterations` times, collecting timing.
    /// When `validate` is true, also compare results against reference files.
    /// When `print_results` is true, print each query's result after execution.
    Run {
        iterations: usize,
        validate: bool,
        print_results: bool,
    },
    /// Prepend `EXPLAIN` to each query, print the result, skip timing.
    Explain,
    /// Run each query once and compare results against reference files.
    Validate,
}

/// Trait implemented by engine-specific query results so the runner can
/// extract row counts (for validation in Run mode) and display text
/// (for Explain mode).
pub trait BenchmarkQueryResult {
    /// Number of result rows (used for row-count validation).
    fn row_count(&self) -> usize;
    /// Human-readable representation of the result (used by Explain mode).
    fn display(self) -> String;
    /// Normalized result for cross-engine validation.
    ///
    /// Returns column names and rows of normalized string values suitable for
    /// comparison across different query engines. Values are normalized using
    /// sqllogictest conventions (floats rounded to 12 decimal places, etc.)
    /// via [`crate::validation`].
    fn normalized_result(&self) -> (Vec<String>, Vec<Vec<String>>);
}
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
    expected_results_dir: Option<PathBuf>,
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
            expected_results_dir: benchmark.expected_results_dir(),
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
    /// Returns the result from the last iteration so callers can validate it.
    fn run_query<R, F>(
        &mut self,
        query_idx: usize,
        format: Format,
        iterations: usize,
        mut f: F,
    ) -> R
    where
        R: BenchmarkQueryResult,
        F: FnMut() -> (Option<Duration>, R),
    {
        self.start_query();

        let mut runs = Vec::with_capacity(iterations);
        let mut last_result: Option<R> = None;

        for _ in 0..iterations {
            let start = Instant::now();
            let (timing, result) = f();
            let elapsed = timing.unwrap_or_else(|| start.elapsed());
            runs.push(elapsed);
            last_result = Some(result);
        }

        let result = last_result.expect("iterations must be > 0");
        let row_count = result.row_count();
        self.record_query(query_idx, format, runs, row_count);
        result
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

    /// Get the path for a `.slt.no` reference result file.
    ///
    /// The path includes an engine-specific subdirectory so each engine can have
    /// its own expected results: `{expected_results_dir}/{engine}/q{idx:02}.slt.no`.
    fn reference_path(&self, query_idx: usize) -> Option<PathBuf> {
        self.expected_results_dir.as_ref().map(|dir| {
            dir.join(self.engine.to_string())
                .join(format!("q{query_idx:02}.slt.no"))
        })
    }

    /// Validate a query result against its `.slt.no` reference file.
    ///
    /// Uses `sqllogictest::default_validator` with `value_normalizer` to compare
    /// actual rows against the expected rows parsed from the slt file.
    ///
    /// Returns `true` if the result matches (or no reference exists), `false` on mismatch.
    fn validate_query_result(&self, query_idx: usize, actual_rows: &mut [Vec<String>]) -> bool {
        let Some(path) = self.reference_path(query_idx) else {
            eprintln!("No expected_results_dir configured, skipping validation for q{query_idx}");
            return true;
        };

        if !path.exists() {
            eprintln!(
                "Reference file {} does not exist, skipping validation for q{query_idx}. \
                 Run with --print-results and redirect output to regenerate it.",
                path.display()
            );
            return true;
        }

        match validate_against_slt(&path, actual_rows) {
            Ok(()) => true,
            Err(msg) => {
                eprintln!(
                    "=== Result mismatch for q{query_idx} ({engine}) ===",
                    engine = self.engine
                );
                eprintln!("{msg}");
                false
            }
        }
    }

    /// Run (or explain) all queries for all formats synchronously.
    ///
    /// In `Run` mode, executes each query `iterations` times, collecting timing.
    /// In `Explain` mode, prepends `EXPLAIN` to each query, executes once, and
    /// prints `R::display()`. No progress bar or timing in Explain mode.
    ///
    /// The `execute` callback returns `(Option<Duration>, R)` where
    /// `Option<Duration>` overrides wall-clock timing, and `R` implements
    /// `BenchmarkQueryResult`.
    pub fn run_all<Ctx, R, S, E>(
        &mut self,
        queries: &[(usize, String)],
        mode: BenchmarkMode,
        mut setup: S,
        mut execute: E,
    ) -> anyhow::Result<()>
    where
        R: BenchmarkQueryResult,
        S: FnMut(Format) -> anyhow::Result<Ctx>,
        E: FnMut(&mut Ctx, usize, Format, &str) -> anyhow::Result<(Option<Duration>, R)>,
    {
        match mode {
            BenchmarkMode::Run {
                iterations,
                validate,
                print_results,
            } => {
                let bar_length = queries.len() * self.formats.len();
                let progress_bar = if self.hide_progress_bar || bar_length == 0 {
                    ProgressBar::hidden()
                } else {
                    ProgressBar::new(bar_length as u64)
                };

                let mut validation_failures = Vec::new();

                for format in self.formats.clone() {
                    let mut ctx = setup(format)?;

                    for (query_idx, query) in queries.iter() {
                        let query_idx = *query_idx;
                        tracing::debug!(%format, query_idx, "Running query");
                        let result = self.run_query(query_idx, format, iterations, || {
                            execute(&mut ctx, query_idx, format, query.as_str()).unwrap_or_else(
                                |err| {
                                    vortex_panic!("query {query_idx} failed: {err}");
                                },
                            )
                        });

                        // Validate the last iteration's result against the reference file.
                        if validate && self.expected_results_dir.is_some() {
                            let (_cols, mut rows) = result.normalized_result();
                            if !self.validate_query_result(query_idx, &mut rows) {
                                validation_failures.push((query_idx, format));
                            }
                        }

                        if print_results {
                            println!("=== Q{query_idx} ===");
                            println!("{}", result.display());
                            println!();
                        }

                        progress_bar.inc(1);
                    }
                }

                progress_bar.finish();

                if !validation_failures.is_empty() {
                    let failed: Vec<String> = validation_failures
                        .iter()
                        .map(|(q, f)| format!("q{q}:{f}"))
                        .collect();
                    anyhow::bail!(
                        "Result validation failed for {engine}: {failed}",
                        engine = self.engine,
                        failed = failed.join(", "),
                    );
                }
            }
            BenchmarkMode::Explain => {
                for format in self.formats.clone() {
                    let mut ctx = setup(format)?;

                    for (query_idx, query) in queries.iter() {
                        let explain_query = format!("EXPLAIN {query}");
                        let (_, result) = execute(&mut ctx, *query_idx, format, &explain_query)?;
                        println!("=== Q{query_idx} [{format}] ===");
                        println!("{query}");
                        println!();
                        println!("{}", result.display());
                        println!();
                    }
                }
            }
            BenchmarkMode::Validate => {
                let mut all_passed = true;
                // Validate using only the first format (results should be format-independent)
                let format = *self.formats.first().ok_or_else(|| {
                    anyhow::anyhow!("At least one format is required for validation")
                })?;
                let mut ctx = setup(format)?;

                for (query_idx, query) in queries.iter() {
                    let (_, result) = execute(&mut ctx, *query_idx, format, query.as_str())?;
                    let (_cols, mut rows) = result.normalized_result();
                    if !self.validate_query_result(*query_idx, &mut rows) {
                        all_passed = false;
                    }
                }

                if !all_passed {
                    anyhow::bail!(
                        "Result validation failed for one or more queries ({engine})",
                        engine = self.engine,
                    );
                }
            }
        }

        Ok(())
    }

    /// Run (or explain) all queries for all formats asynchronously.
    ///
    /// Same semantics as `run_all` but for async execute callbacks.
    /// Use `Box::pin(async move { ... })` in the closure.
    pub async fn run_all_async<Ctx, R, S, SFut, E>(
        &mut self,
        queries: &[(usize, String)],
        mode: BenchmarkMode,
        setup: S,
        mut execute: E,
    ) -> anyhow::Result<()>
    where
        R: BenchmarkQueryResult,
        S: Fn(Format) -> SFut,
        SFut: Future<Output = anyhow::Result<Ctx>>,
        E: for<'c> FnMut(
            usize,
            &'c Ctx,
            &'c str,
        ) -> Pin<
            Box<dyn Future<Output = anyhow::Result<(Option<Duration>, R)>> + 'c>,
        >,
    {
        match mode {
            BenchmarkMode::Run {
                iterations,
                validate,
                print_results,
            } => {
                let bar_length = queries.len() * self.formats.len();
                let progress_bar = if self.hide_progress_bar || bar_length == 0 {
                    ProgressBar::hidden()
                } else {
                    ProgressBar::new(bar_length as u64)
                };

                let mut validation_failures = Vec::new();

                for format in self.formats.clone() {
                    let ctx = setup(format).await?;

                    for (query_idx, query) in queries.iter() {
                        let query_idx = *query_idx;

                        self.start_query();

                        let mut runs = Vec::with_capacity(iterations);
                        let mut last_result: Option<R> = None;

                        tracing::debug!(%format, query_idx, "Running query");

                        for _ in 0..iterations {
                            let start = Instant::now();
                            let (timing, result) = execute(query_idx, &ctx, query.as_str())
                                .await
                                .unwrap_or_else(|err| {
                                    vortex_panic!("query {query_idx} failed: {err}");
                                });
                            let elapsed = timing.unwrap_or_else(|| start.elapsed());
                            runs.push(elapsed);
                            last_result = Some(result);
                        }

                        let result = last_result.expect("iterations must be > 0");
                        let row_count = result.row_count();
                        self.record_query(query_idx, format, runs, row_count);

                        // Validate the last iteration's result against the reference file.
                        if validate && self.expected_results_dir.is_some() {
                            let (_cols, mut rows) = result.normalized_result();
                            if !self.validate_query_result(query_idx, &mut rows) {
                                validation_failures.push((query_idx, format));
                            }
                        }

                        if print_results {
                            println!("=== Q{query_idx} ===");
                            println!("{}", result.display());
                            println!();
                        }

                        progress_bar.inc(1);
                    }
                }

                progress_bar.finish();

                if !validation_failures.is_empty() {
                    let failed: Vec<String> = validation_failures
                        .iter()
                        .map(|(q, f)| format!("q{q}:{f}"))
                        .collect();
                    anyhow::bail!(
                        "Result validation failed for {engine}: {failed}",
                        engine = self.engine,
                        failed = failed.join(", "),
                    );
                }
            }
            BenchmarkMode::Explain => {
                for format in self.formats.clone() {
                    let ctx = setup(format).await?;

                    for (query_idx, query) in queries.iter() {
                        let explain_query = format!("EXPLAIN {query}");
                        let (_, result) = execute(*query_idx, &ctx, &explain_query).await?;
                        println!("=== Q{query_idx} [{format}] ===");
                        println!("{query}");
                        println!();
                        println!("{}", result.display());
                        println!();
                    }
                }
            }
            BenchmarkMode::Validate => {
                let mut all_passed = true;
                let format = *self.formats.first().ok_or_else(|| {
                    anyhow::anyhow!("At least one format is required for validation")
                })?;
                let ctx = setup(format).await?;

                for (query_idx, query) in queries.iter() {
                    let (_, result) = execute(*query_idx, &ctx, query.as_str()).await?;
                    let (_cols, mut rows) = result.normalized_result();
                    if !self.validate_query_result(*query_idx, &mut rows) {
                        all_passed = false;
                    }
                }

                if !all_passed {
                    anyhow::bail!(
                        "Result validation failed for one or more queries ({engine})",
                        engine = self.engine,
                    );
                }
            }
        }

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
