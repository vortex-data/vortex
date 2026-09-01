// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The E1 evaluation: the morsel executor against the V1 `LayoutReader`.
//!
//! The fair contract, in order:
//!
//! 1. Build the fixture once. Both executors read the same in-memory segments.
//! 2. Validate every executor's output against V1's — same row count, same ordered content —
//!    *before* anything is timed. A configuration that disagrees is reported as a failure and
//!    never appears in the timing table.
//! 3. Warm up once per executor, then run five alternating iterations so drift in machine state
//!    hits both rows equally. Report the median.
//!
//! Run with: `cargo run --release -p vortex-morsel --features _test-harness --bin morsel-eval`

use std::sync::Arc;
use std::time::Duration;

use vortex_array::array_session;
use vortex_error::VortexResult;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSession;
use vortex_layout::LayoutRef;
use vortex_layout::segments::SegmentSource;
use vortex_layout::session::LayoutSession;
use vortex_morsel::fixtures::Fixture;
use vortex_morsel::fixtures::write_fixture;
use vortex_morsel::harness::MorselConfig;
use vortex_morsel::harness::Query;
use vortex_morsel::harness::RunOutcome;
use vortex_morsel::harness::assert_same_rows;
use vortex_morsel::harness::run_morsel;
use vortex_morsel::harness::run_v1;
use vortex_morsel::harness::run_v1_tokio;
use vortex_morsel::nodes::ConjunctMode;
use vortex_morsel::workloads;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

const ITERATIONS: usize = 5;

/// The executor configurations in the matrix.
#[derive(Clone, Copy)]
enum Row {
    /// V1 `LayoutReader`, single-threaded. The oracle and the apples-to-apples baseline.
    V1Single,
    /// V1 `LayoutReader` on a multi-threaded Tokio runtime, the way DataFusion drives it.
    V1Tokio(usize),
    /// The morsel executor.
    Morsel(MorselConfig),
}

impl Row {
    fn label(&self) -> String {
        match self {
            Row::V1Single => "A  V1 (1 thread)".to_string(),
            Row::V1Tokio(threads) => format!("A' V1 (tokio x{threads})"),
            Row::Morsel(config) => {
                let mode = match config.mode {
                    ConjunctMode::Cascade => "",
                    ConjunctMode::Parallel => ", parallel",
                };
                let morsel = if config.morsel_rows == 0 {
                    "splits".to_string()
                } else {
                    format!("{}r", config.morsel_rows)
                };
                let reuse = if config.share_decodes {
                    ""
                } else {
                    ", no-reuse"
                };
                format!("D  morsel (x{}, {morsel}{mode}{reuse})", config.threads)
            }
        }
    }
}

struct Timing {
    label: String,
    median: Duration,
    rows: usize,
    ttfb: Option<Duration>,
    requests: Option<u64>,
    decodes: Option<u64>,
    reuses: Option<u64>,
    io_uses: Option<u64>,
    morsels: Option<u64>,
}

fn main() -> VortexResult<()> {
    let session = array_session()
        .with::<LayoutSession>()
        .with::<RuntimeSession>();
    let threads = get_available_parallelism().unwrap_or(4);
    let runtime = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(threads)
        .enable_all()
        .build()
        .map_err(|err| vortex_error::vortex_err!("failed to build the tokio runtime: {err}"))?;

    let scale: usize = std::env::var("MORSEL_EVAL_ROWS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(1_000_000);

    println!("# Morsel executor evaluation (E1)");
    println!();
    println!(
        "host: {threads} logical cores; segments in memory; {scale} rows per workload; \
         {ITERATIONS} alternating iterations, median reported"
    );
    println!("both executors use workers prepared outside the timed interval");
    println!();

    let workloads = vec![
        workloads::string_heavy(scale / 4),
        workloads::wide_numeric(scale),
        workloads::narrow_analytic(scale),
    ];

    let mut failures = Vec::new();

    for workload in workloads {
        let fixture =
            block_on(|_handle| async { write_fixture(workload.columns, &session).await })?;
        let segments: Arc<dyn SegmentSource> = Arc::clone(&fixture.segments);

        println!("## {} — {}", workload.name, workload.shape);
        println!();
        println!(
            "{} rows, {} natural splits",
            fixture.row_count,
            natural_splits(&fixture, &workload.queries[0])?
        );
        println!();

        for query in &workload.queries {
            let rows_config = [
                Row::V1Single,
                Row::V1Tokio(threads),
                Row::Morsel(MorselConfig {
                    threads: 1,
                    ..Default::default()
                }),
                Row::Morsel(MorselConfig {
                    threads: 1,
                    share_decodes: false,
                    ..Default::default()
                }),
                Row::Morsel(MorselConfig {
                    threads,
                    ..Default::default()
                }),
                Row::Morsel(MorselConfig {
                    threads,
                    morsel_rows: 65_536,
                    ..Default::default()
                }),
                Row::Morsel(MorselConfig {
                    threads,
                    mode: ConjunctMode::Parallel,
                    ..Default::default()
                }),
            ];

            // Step 1: the oracle. Every row must agree with V1 before any timing happens.
            let oracle = run_v1(&session, &fixture.layout, &segments, query)?;
            let dtype = query
                .projection
                .bind(fixture.layout.dtype())?
                .dtype()
                .clone();
            let mut validated = Vec::new();
            for row in rows_config {
                let outcome = run_once(&runtime, &session, &fixture.layout, &segments, query, row)?;
                match assert_same_rows(&session, &dtype, &oracle, &outcome) {
                    Ok(()) => validated.push(row),
                    Err(err) => {
                        failures.push(format!(
                            "{} / {} / {}: {err}",
                            workload.name,
                            query.name,
                            row.label()
                        ));
                    }
                }
            }

            // Step 2: alternating iterations over the validated rows.
            let mut samples: Vec<Vec<RunOutcome>> = validated.iter().map(|_| Vec::new()).collect();
            for _ in 0..ITERATIONS {
                for (idx, row) in validated.iter().enumerate() {
                    let outcome =
                        run_once(&runtime, &session, &fixture.layout, &segments, query, *row)?;
                    samples[idx].push(outcome);
                }
            }

            let timings: Vec<Timing> = validated
                .iter()
                .zip(samples)
                .map(|(row, mut runs)| {
                    runs.sort_by_key(|run| run.wall);
                    let median = &runs[runs.len() / 2];
                    Timing {
                        label: row.label(),
                        median: median.wall,
                        rows: median.rows,
                        ttfb: median.time_to_first_batch,
                        requests: median.stats.as_ref().map(|s| s.io_requests),
                        decodes: median.stats.as_ref().map(|s| s.decodes),
                        reuses: median.stats.as_ref().map(|s| s.decode_reuses),
                        io_uses: median.stats.as_ref().map(|s| s.io_uses),
                        morsels: median.stats.as_ref().map(|s| s.morsels),
                    }
                })
                .collect();

            report(query, &timings);
        }
    }

    if failures.is_empty() {
        println!("All configurations matched the V1 oracle.");
        Ok(())
    } else {
        println!("## Oracle failures");
        println!();
        for failure in &failures {
            println!("- {failure}");
        }
        vortex_error::vortex_bail!("{} configurations disagreed with V1", failures.len())
    }
}

fn run_once(
    runtime: &tokio::runtime::Runtime,
    session: &VortexSession,
    layout: &LayoutRef,
    segments: &Arc<dyn SegmentSource>,
    query: &Query,
    row: Row,
) -> VortexResult<RunOutcome> {
    match row {
        Row::V1Single => run_v1(session, layout, segments, query),
        Row::V1Tokio(_) => run_v1_tokio(runtime, session, layout, segments, query),
        Row::Morsel(config) => run_morsel(session, layout, segments, query, config),
    }
}

fn natural_splits(fixture: &Fixture, query: &Query) -> VortexResult<usize> {
    let plan = vortex_morsel::build_plan(
        &fixture.layout,
        &query.projection,
        query.filter.as_ref(),
        ConjunctMode::Cascade,
    )?;
    Ok(plan.natural_splits().len())
}

fn report(query: &Query, timings: &[Timing]) {
    let baseline = timings
        .iter()
        .find(|t| t.label.starts_with("A  "))
        .map(|t| t.median)
        .unwrap_or_default();

    println!("### {}", query.name);
    println!();
    println!(
        "| executor | wall | vs V1 | rows | ttfb | morsels | uses | reqs | decodes | reuses |"
    );
    println!("|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|");
    for timing in timings {
        let ratio = if baseline.is_zero() {
            "—".to_string()
        } else {
            format!(
                "{:.2}x",
                timing.median.as_secs_f64() / baseline.as_secs_f64()
            )
        };
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} | {} | {} |",
            timing.label,
            millis(timing.median),
            ratio,
            timing.rows,
            timing.ttfb.map(millis).unwrap_or_else(|| "—".to_string()),
            opt(timing.morsels),
            opt(timing.io_uses),
            opt(timing.requests),
            opt(timing.decodes),
            opt(timing.reuses),
        );
    }
    println!();
}

/// Format a duration in milliseconds, avoiding `Debug` formatting.
fn millis(duration: Duration) -> String {
    format!("{:.3}ms", duration.as_secs_f64() * 1000.0)
}

fn opt(value: Option<u64>) -> String {
    value.map(|v| v.to_string()).unwrap_or_else(|| "—".into())
}
