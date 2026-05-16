//! Bench harness for engine-driven queries.
//!
//! Goal: measure the same shape across queries and across engines.
//! The split mirrors what other engines' benchmark runners do —
//! e.g. `vortex-bench`'s `SqlBenchmarkRunner` calls `setup(format)`
//! once per format and `execute(query)` per iteration; we do
//! `prepare` once per query and `execute` per iteration.
//!
//! Per query, an implementer of [`BenchQuery`] provides:
//!
//! - `prepare(paths)`: untimed setup. Open files, evaluate
//!   file-level prune checks, build any cached graph fragments —
//!   anything a planner would do once and hand to a runtime.
//! - `execute(prepared, workers)`: timed query execution. Builds
//!   the operator graph (cheaply, from the prepared inputs), runs
//!   the task, returns the typed result.
//!
//! [`BenchHarness::run`] runs prepare once, executes `iterations`
//! times, captures per-iteration wall times, and returns a
//! [`BenchReport`] with median / min / max stats.
//!
//! Binaries (e.g. `q20_unioned`) parse CLI args for shards /
//! workers / iterations and hand the rest to the harness — no
//! per-query timing boilerplate to maintain.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::Duration;
use std::time::Instant;

use parking_lot::Mutex;

use crate::EngineResult;
use crate::ExecutionMetrics;
use crate::OperatorGraph;
use crate::PreparedTask;
use crate::TaskOptions;
use crate::drivers::EngineWorkerPool;

/// One benchmarkable query — a typed `prepare` + `execute` pair.
///
/// Prepared state is kept by reference across iterations so a
/// single open-and-prune phase amortises across many query runs.
/// Implementations should make `Prepared` cheap to clone *into*
/// the per-iteration operator graph; for q20-shaped queries,
/// `LazyVortexFile` is `Clone` and that's exactly what
/// `Q20Unioned::execute` does.
pub trait BenchQuery {
    /// State produced once by `prepare` and reused across every
    /// `execute` iteration. For file-scan queries this typically
    /// holds opened file handles + planner-side decisions
    /// (e.g. which shards were file-pruned).
    type Prepared;
    /// Typed query result. The harness keeps the output of the
    /// last iteration so callers can inspect it (e.g. assert match
    /// count) after benchmarking.
    type Output;

    /// Short identifier for printed output. Conventionally lower
    /// snake-case matching the binary name.
    fn name(&self) -> &str;

    /// Open files, evaluate file-level pruning, build cached
    /// graph fragments. Untimed.
    fn prepare(&self, paths: Vec<PathBuf>) -> EngineResult<Self::Prepared>;

    /// Run the query against the prepared state. Timed by the
    /// harness. Should be deterministic — repeated calls with the
    /// same prepared state should produce the same output.
    ///
    /// `runner` owns the worker pool and is shared across every
    /// iteration so the OS-thread cost is paid once per
    /// `BenchHarness::run`, not once per iteration.
    fn execute(
        &self,
        prepared: &Self::Prepared,
        runner: &Runner,
    ) -> EngineResult<Self::Output>;

    /// One-line human summary of an output (e.g. `"matches=4"`).
    /// Printed alongside the per-iteration timings and on the
    /// final summary line.
    fn output_summary(&self, output: &Self::Output) -> String;

    /// Optional per-prepare summary, printed once after the setup
    /// phase. Default empty. Override to surface things like
    /// "pruned 96 / 100 shards at file-stat level".
    fn prepared_summary(&self, _prepared: &Self::Prepared) -> String {
        String::new()
    }
}

/// Result of a benchmark run. Carries the prepare time, every
/// iteration's wall time, and the output of the last iteration.
pub struct BenchReport<O> {
    pub name: String,
    pub paths: usize,
    pub worker_count: usize,
    pub iterations: usize,
    pub setup: Duration,
    pub prepared_summary: String,
    pub runs: Vec<Duration>,
    pub last_output: Option<O>,
}

impl<O> BenchReport<O> {
    /// Per-iteration runs *excluding* the first (which usually
    /// pays cold-cache costs). Returned in original order. If only
    /// one iteration ran, returns it unchanged.
    pub fn warm_runs(&self) -> &[Duration] {
        if self.runs.len() <= 1 {
            &self.runs[..]
        } else {
            &self.runs[1..]
        }
    }

    pub fn warm_median(&self) -> Option<Duration> {
        let mut warm: Vec<Duration> = self.warm_runs().to_vec();
        warm.sort();
        warm.get(warm.len() / 2).copied()
    }

    pub fn warm_min(&self) -> Option<Duration> {
        self.warm_runs().iter().copied().min()
    }

    pub fn warm_max(&self) -> Option<Duration> {
        self.warm_runs().iter().copied().max()
    }
}

impl<O> BenchReport<O> {
    /// Standard summary line. Format is line-stable so it's
    /// scrape-friendly for follow-up tooling.
    ///
    /// Pass the originating [`BenchQuery`] so the report can call
    /// its `output_summary` for the result line. Generic over
    /// `Q::Output = O` so the type checker enforces that the query
    /// matches the report.
    pub fn print_with<Q>(&self, query: &Q)
    where
        Q: BenchQuery<Output = O>,
    {
        eprintln!(
            "[{name}] paths={paths} workers={workers} iterations={iterations}",
            name = self.name,
            paths = self.paths,
            workers = self.worker_count,
            iterations = self.iterations,
        );
        eprintln!("  setup: {:?}", self.setup);
        if !self.prepared_summary.is_empty() {
            eprintln!("  prepared: {}", self.prepared_summary);
        }
        for (i, t) in self.runs.iter().enumerate() {
            let tag = if i == 0 && self.runs.len() > 1 {
                " (warm-up)"
            } else {
                ""
            };
            eprintln!("  iter {i}: {:?}{tag}", t);
        }
        if let Some(out) = self.last_output.as_ref() {
            eprintln!("  result: {}", query.output_summary(out));
        }
        if self.runs.len() > 1
            && let (Some(med), Some(min), Some(max)) =
                (self.warm_median(), self.warm_min(), self.warm_max())
        {
            eprintln!(
                "  warm median: {med:?}    min: {min:?}    max: {max:?}    n={}",
                self.warm_runs().len()
            );
        }
    }
}

/// Owns the OS-thread worker pool used by every iteration of a
/// benchmark run. Created once by [`BenchHarness`]; threaded into
/// each `BenchQuery::execute` call so the per-iteration cost is the
/// graph build + scheduler turns, not the cost of spinning up N
/// fresh OS threads.
///
/// `worker_count == 1` skips the pool entirely and runs synchronously
/// on the calling thread via [`PreparedTask::run`].
pub enum Runner {
    Single,
    Pool(EngineWorkerPool),
}

impl Runner {
    pub fn new(worker_count: usize) -> Self {
        if worker_count > 1 {
            Self::Pool(EngineWorkerPool::new(worker_count))
        } else {
            Self::Single
        }
    }

    pub fn worker_count(&self) -> usize {
        match self {
            Self::Single => 1,
            Self::Pool(p) => p.worker_count(),
        }
    }

    /// Run `graph` to completion. Builds default `TaskOptions`
    /// matching the runner's worker count and drops the resulting
    /// metrics. Use [`Runner::run_with_metrics`] if you need to
    /// inspect them.
    pub fn run(&self, graph: OperatorGraph) -> EngineResult<()> {
        let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
        self.run_with_metrics(graph, metrics)
    }

    pub fn run_with_metrics(
        &self,
        graph: OperatorGraph,
        metrics: Arc<Mutex<ExecutionMetrics>>,
    ) -> EngineResult<()> {
        let options = TaskOptions {
            worker_count: self.worker_count(),
            ..TaskOptions::default()
        };
        match self {
            Self::Single => {
                PreparedTask::prepare(graph, metrics, options)?.run()?;
                Ok(())
            }
            Self::Pool(p) => p.run_graph(graph, metrics, options),
        }
    }
}

/// Driver for [`BenchQuery`] runs. Owns the worker pool for the
/// duration of one `run` invocation so iteration N sees the same
/// pool iteration N-1 used.
pub struct BenchHarness;

impl BenchHarness {
    /// Run `query` over `paths` for `iterations` iterations at
    /// `worker_count` workers. Setup is timed once and excluded
    /// from per-iteration wall times. The pool is created once and
    /// reused across every iteration.
    pub fn run<Q: BenchQuery>(
        query: &Q,
        paths: Vec<PathBuf>,
        worker_count: usize,
        iterations: usize,
    ) -> EngineResult<BenchReport<Q::Output>> {
        assert!(iterations > 0, "iterations must be positive");
        let path_count = paths.len();
        let setup_start = Instant::now();
        let prepared = query.prepare(paths)?;
        let setup = setup_start.elapsed();
        let prepared_summary = query.prepared_summary(&prepared);

        let runner = Runner::new(worker_count);

        let mut runs = Vec::with_capacity(iterations);
        let mut last_output = None;
        for _ in 0..iterations {
            let start = Instant::now();
            let output = query.execute(&prepared, &runner)?;
            runs.push(start.elapsed());
            last_output = Some(output);
        }

        Ok(BenchReport {
            name: query.name().to_string(),
            paths: path_count,
            worker_count,
            iterations,
            setup,
            prepared_summary,
            runs,
            last_output,
        })
    }

    /// Convenience: run + print summary in one call. Used by
    /// binaries that don't need the structured report.
    pub fn run_and_print<Q: BenchQuery>(
        query: &Q,
        paths: Vec<PathBuf>,
        worker_count: usize,
        iterations: usize,
    ) -> EngineResult<BenchReport<Q::Output>> {
        let report = Self::run(query, paths, worker_count, iterations)?;
        report.print_with(query);
        Ok(report)
    }
}

/// Read every Vortex file in a directory, sort lexicographically,
/// truncate to at most `limit` entries. Convenience for binaries
/// that just want to point at a clickbench partition directory.
pub fn vortex_files_in(dir: &str, limit: usize) -> Vec<PathBuf> {
    let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
        .expect("readdir")
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "vortex"))
        .collect();
    paths.sort();
    paths.truncate(limit);
    paths
}
