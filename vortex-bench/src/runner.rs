// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Generic benchmark runner infrastructure to reduce boilerplate across engine-specific benchmarks.

use std::fs::File;
use std::future::Future;
use std::io::Write;
use std::path::Path;
use std::path::PathBuf;
use std::pin::Pin;
use std::time::Duration;
use std::time::Instant;

use datafusion_sqllogictest::value_normalizer;
use indicatif::ProgressBar;
use sqllogictest::Condition;
use sqllogictest::DefaultColumnType;
use sqllogictest::QueryExpect;
use sqllogictest::Record;
use sqllogictest::default_validator;
use sqllogictest::parse_file;
use vortex::error::vortex_panic;

use crate::Benchmark;
use crate::BenchmarkDataset;
use crate::Engine;
use crate::Format;
use crate::Target;
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
}

/// Trait implemented by engine-specific query results so the runner can
/// extract row counts (for validation in Run mode) and display text
/// (for Explain mode).
pub trait BenchmarkQueryResult {
    /// Number of result rows (used for row-count validation).
    fn row_count(&self) -> usize;
    /// Human-readable representation of the result (used by Explain mode).
    fn display(self) -> String;
    /// Raw result rows for validation.
    ///
    /// Returns column names and rows of string values extracted from the
    /// query result. No cross-engine normalization is applied.
    fn result_rows(&self) -> (Vec<String>, Vec<Vec<String>>);
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
                validate: _,
                print_results,
            } => {
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
                        let result = self.run_query(query_idx, format, iterations, || {
                            execute(&mut ctx, query_idx, format, query.as_str()).unwrap_or_else(
                                |err| {
                                    vortex_panic!("query {query_idx} failed: {err}");
                                },
                            )
                        });

                        if print_results {
                            println!("=== Q{query_idx} ===");
                            println!("{}", result.display());
                            println!();
                        }

                        progress_bar.inc(1);
                    }
                }

                progress_bar.finish();
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
                validate: _,
                print_results,
            } => {
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

                        if print_results {
                            println!("=== Q{query_idx} ===");
                            println!("{}", result.display());
                            println!();
                        }

                        progress_bar.inc(1);
                    }
                }

                progress_bar.finish();
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
        }

        Ok(())
    }

    /// Run benchmarks driven by a consolidated `.slt` file.
    ///
    /// Parses the SLT file, filters records by `engine_label` (via `onlyif`/`skipif`
    /// conditions), and for each matching `query` record with a `bench_N` label:
    /// executes `iterations` times, validates against expected results, and records
    /// timing measurements.
    ///
    /// `Statement` records are executed once as setup (not timed).
    #[allow(clippy::too_many_arguments)]
    pub fn run_slt<R, S, E>(
        &mut self,
        slt_path: &Path,
        engine_label: &str,
        format: Format,
        iterations: usize,
        validate: bool,
        include_queries: Option<&Vec<usize>>,
        exclude_queries: Option<&Vec<usize>>,
        mut setup: S,
        mut execute: E,
    ) -> anyhow::Result<()>
    where
        R: BenchmarkQueryResult,
        S: FnMut(&str) -> anyhow::Result<()>,
        E: FnMut(&str) -> anyhow::Result<(Option<Duration>, R)>,
    {
        let records = parse_file::<DefaultColumnType>(slt_path)
            .map_err(|e| anyhow::anyhow!("Failed to parse {}: {e}", slt_path.display()))?;

        let bench_records = collect_bench_records(&records, engine_label);

        let progress_bar = if self.hide_progress_bar || bench_records.is_empty() {
            ProgressBar::hidden()
        } else {
            ProgressBar::new(bench_records.len() as u64)
        };

        let mut validation_failures = Vec::new();

        for record in &records {
            match record {
                Record::Statement {
                    conditions, sql, ..
                } if !should_skip(conditions, engine_label) => {
                    setup(sql)?;
                }
                Record::Query {
                    conditions,
                    sql,
                    expected,
                    ..
                } if !should_skip(conditions, engine_label) => {
                    let (query_idx, expected_results) =
                        match extract_bench_query(expected, include_queries, exclude_queries) {
                            Some(v) => v,
                            None => continue,
                        };

                    let result = self.run_query(query_idx, format, iterations, || {
                        execute(sql).unwrap_or_else(|err| {
                            vortex_panic!("query {query_idx} failed: {err}");
                        })
                    });

                    if validate {
                        let (_cols, mut rows) = result.result_rows();
                        rows.sort();
                        if !default_validator(value_normalizer, &rows, expected_results) {
                            let actual_flat: Vec<String> =
                                rows.iter().map(|row| row.join(" ")).collect();
                            eprintln!(
                                "=== Result mismatch for bench_{query_idx} ({engine}) ===",
                                engine = self.engine
                            );
                            print_validation_diff(expected_results, &actual_flat);
                            validation_failures.push(query_idx);
                        }
                    }

                    progress_bar.inc(1);
                }
                _ => {}
            }
        }

        progress_bar.finish();

        if !validation_failures.is_empty() {
            let failed: Vec<String> = validation_failures
                .iter()
                .map(|q| format!("bench_{q}"))
                .collect();
            anyhow::bail!(
                "SLT validation failed for {engine}: {failed}",
                engine = self.engine,
                failed = failed.join(", "),
            );
        }

        Ok(())
    }

    /// Async version of [`Self::run_slt`].
    #[allow(clippy::too_many_arguments)]
    pub async fn run_slt_async<R, S, SFut, E>(
        &mut self,
        slt_path: &Path,
        engine_label: &str,
        format: Format,
        iterations: usize,
        validate: bool,
        include_queries: Option<&Vec<usize>>,
        exclude_queries: Option<&Vec<usize>>,
        setup: S,
        mut execute: E,
    ) -> anyhow::Result<()>
    where
        R: BenchmarkQueryResult,
        S: Fn(&str) -> SFut,
        SFut: Future<Output = anyhow::Result<()>>,
        E: for<'a> FnMut(
            &'a str,
        ) -> Pin<
            Box<dyn Future<Output = anyhow::Result<(Option<Duration>, R)>> + 'a>,
        >,
    {
        let records = parse_file::<DefaultColumnType>(slt_path)
            .map_err(|e| anyhow::anyhow!("Failed to parse {}: {e}", slt_path.display()))?;

        let bench_records = collect_bench_records(&records, engine_label);

        let progress_bar = if self.hide_progress_bar || bench_records.is_empty() {
            ProgressBar::hidden()
        } else {
            ProgressBar::new(bench_records.len() as u64)
        };

        let mut validation_failures = Vec::new();

        for record in &records {
            match record {
                Record::Statement {
                    conditions, sql, ..
                } if !should_skip(conditions, engine_label) => {
                    setup(sql).await?;
                }
                Record::Query {
                    conditions,
                    sql,
                    expected,
                    ..
                } if !should_skip(conditions, engine_label) => {
                    let (query_idx, expected_results) =
                        match extract_bench_query(expected, include_queries, exclude_queries) {
                            Some(v) => v,
                            None => continue,
                        };

                    self.start_query();

                    let mut runs = Vec::with_capacity(iterations);
                    let mut last_result: Option<R> = None;

                    for _ in 0..iterations {
                        let start = Instant::now();
                        let (timing, result) = execute(sql).await.unwrap_or_else(|err| {
                            vortex_panic!("query {query_idx} failed: {err}");
                        });
                        let elapsed = timing.unwrap_or_else(|| start.elapsed());
                        runs.push(elapsed);
                        last_result = Some(result);
                    }

                    let result = last_result.expect("iterations must be > 0");
                    let row_count = result.row_count();
                    self.record_query(query_idx, format, runs, row_count);

                    if validate {
                        let (_cols, mut rows) = result.result_rows();
                        rows.sort();
                        if !default_validator(value_normalizer, &rows, expected_results) {
                            let actual_flat: Vec<String> =
                                rows.iter().map(|row| row.join(" ")).collect();
                            eprintln!(
                                "=== Result mismatch for bench_{query_idx} ({engine}) ===",
                                engine = self.engine
                            );
                            print_validation_diff(expected_results, &actual_flat);
                            validation_failures.push(query_idx);
                        }
                    }

                    progress_bar.inc(1);
                }
                _ => {}
            }
        }

        progress_bar.finish();

        if !validation_failures.is_empty() {
            let failed: Vec<String> = validation_failures
                .iter()
                .map(|q| format!("bench_{q}"))
                .collect();
            anyhow::bail!(
                "SLT validation failed for {engine}: {failed}",
                engine = self.engine,
                failed = failed.join(", "),
            );
        }

        Ok(())
    }
}

/// Check whether a record should be skipped for the given engine label.
fn should_skip(conditions: &[Condition], engine_label: &str) -> bool {
    for cond in conditions {
        match cond {
            Condition::OnlyIf { label } => {
                if label != engine_label {
                    return true;
                }
            }
            Condition::SkipIf { label } => {
                if label == engine_label {
                    return true;
                }
            }
        }
    }
    false
}

/// Extract the query index and expected results from a `bench_N` labeled query record.
///
/// Returns `None` if the record doesn't have a `bench_` label or is filtered out
/// by `include_queries`.
fn extract_bench_query<'a>(
    expected: &'a QueryExpect<DefaultColumnType>,
    include_queries: Option<&Vec<usize>>,
    exclude_queries: Option<&Vec<usize>>,
) -> Option<(usize, &'a Vec<String>)> {
    let QueryExpect::Results {
        label: Some(label),
        results,
        ..
    } = expected
    else {
        return None;
    };

    let name = label.strip_prefix("bench_")?;
    let query_idx: usize = name.parse().ok()?;

    if let Some(included) = include_queries
        && !included.contains(&query_idx)
    {
        return None;
    }

    if let Some(excluded) = exclude_queries
        && excluded.contains(&query_idx)
    {
        return None;
    }

    Some((query_idx, results))
}

/// Count the bench query records that match the engine label and will be executed.
fn collect_bench_records(records: &[Record<DefaultColumnType>], engine_label: &str) -> Vec<usize> {
    records
        .iter()
        .filter_map(|record| {
            if let Record::Query {
                conditions,
                expected,
                ..
            } = record
            {
                if should_skip(conditions, engine_label) {
                    return None;
                }
                if let QueryExpect::Results {
                    label: Some(label), ..
                } = expected
                    && let Some(name) = label.strip_prefix("bench_")
                {
                    return name.parse::<usize>().ok();
                }
            }
            None
        })
        .collect()
}

/// Print a human-readable diff between expected and actual results.
fn print_validation_diff(expected: &[String], actual: &[String]) {
    eprintln!(
        "Expected {} lines, got {} lines",
        expected.len(),
        actual.len()
    );

    let max_lines = expected.len().max(actual.len()).min(20);
    for i in 0..max_lines {
        let exp = expected.get(i).map(String::as_str).unwrap_or("<missing>");
        let act = actual.get(i).map(String::as_str).unwrap_or("<missing>");
        if exp != act {
            eprintln!("  line {i}: expected: {exp}");
            eprintln!("  line {i}:   actual: {act}");
        }
    }
    if expected.len().max(actual.len()) > 20 {
        eprintln!("  ... (truncated)");
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
