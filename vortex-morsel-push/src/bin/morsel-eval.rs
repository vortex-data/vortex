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
//! 3. Warm up and sample each in-memory executor independently. Report median and min/max so
//!    scheduler noise remains visible.
//!
//! Run with:
//! `cargo run --release -p vortex-morsel-push --features _test-harness --bin morsel-push-eval`

use std::sync::Arc;
use std::time::Duration;

use vortex_array::array_session;
use vortex_error::VortexResult;
use vortex_io::runtime::single::block_on;
use vortex_io::session::RuntimeSession;
use vortex_layout::LayoutRef;
use vortex_layout::segments::SegmentSource;
use vortex_layout::session::LayoutSession;
use vortex_morsel_push::fixtures::Fixture;
use vortex_morsel_push::fixtures::write_fixture;
use vortex_morsel_push::harness::MorselConfig;
use vortex_morsel_push::harness::Query;
use vortex_morsel_push::harness::RunOutcome;
use vortex_morsel_push::harness::assert_same_rows;
use vortex_morsel_push::harness::run_morsel;
use vortex_morsel_push::harness::run_v1;
use vortex_morsel_push::harness::run_v1_tokio;
use vortex_morsel_push::nodes::ConjunctMode;
use vortex_morsel_push::workloads;
use vortex_session::VortexSession;
use vortex_utils::parallelism::get_available_parallelism;

const DEFAULT_ITERATIONS: usize = 5;

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
    min: Duration,
    max: Duration,
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
    let fineweb_rows = env_rows("MORSEL_FINEWEB_ROWS")?.unwrap_or(scale / 4);
    let clickbench_rows = env_rows("MORSEL_CLICKBENCH_ROWS")?.unwrap_or(scale);
    let iterations = env_rows("MORSEL_EVAL_ITERATIONS")?
        .unwrap_or(DEFAULT_ITERATIONS)
        .max(1);
    let selected_morsel_rows = std::env::var("MORSEL_EVAL_MORSEL_ROWS")
        .ok()
        .map(|value| parse_row_sizes("MORSEL_EVAL_MORSEL_ROWS", &value))
        .transpose()?;
    let selected_workload = std::env::var("MORSEL_EVAL_WORKLOAD").ok();

    println!("# Morsel executor evaluation (E1)");
    println!();
    println!(
        "host: {threads} logical cores; segments in memory; fineweb-shaped={fineweb_rows} rows, \
         clickbench-shaped={clickbench_rows} rows; {iterations} grouped iterations, median reported"
    );
    println!("both executors use workers prepared outside the timed interval");
    println!();

    let mut workload_set = Vec::new();
    if selected_workload
        .as_deref()
        .is_none_or(|name| name == "string-heavy")
    {
        workload_set.push(workloads::string_heavy(fineweb_rows));
    }
    if selected_workload
        .as_deref()
        .is_none_or(|name| name == "wide-numeric")
    {
        workload_set.push(workloads::wide_numeric(clickbench_rows));
    }
    if selected_workload
        .as_deref()
        .is_none_or(|name| name == "narrow-analytic")
    {
        workload_set.push(workloads::narrow_analytic(scale));
    }
    if workload_set.is_empty() {
        vortex_error::vortex_bail!(
            "MORSEL_EVAL_WORKLOAD must be string-heavy, wide-numeric, or narrow-analytic"
        );
    }

    let mut failures = Vec::new();

    for workload in workload_set {
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
            let rows_config: Vec<Row> = match &selected_morsel_rows {
                Some(sizes) => sizes
                    .iter()
                    .copied()
                    .map(|morsel_rows| {
                        Row::Morsel(MorselConfig {
                            threads,
                            morsel_rows,
                            ..Default::default()
                        })
                    })
                    .collect(),
                None => vec![
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
                ],
            };

            // Step 1: the oracle. Every row must agree with V1 before any timing happens.
            let oracle = run_v1(&session, &fixture.layout, &segments, query)?;
            let dtype = query
                .projection
                .bind(fixture.layout.dtype())?
                .dtype()
                .clone();
            let mut validated = Vec::new();
            for &row in &rows_config {
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

            // Step 2: independently warm and sample each in-memory configuration.
            let mut samples: Vec<Vec<RunOutcome>> = validated.iter().map(|_| Vec::new()).collect();
            for (idx, row) in validated.iter().enumerate() {
                drop(run_once(
                    &runtime,
                    &session,
                    &fixture.layout,
                    &segments,
                    query,
                    *row,
                )?);
                for _ in 0..iterations {
                    let mut outcome =
                        run_once(&runtime, &session, &fixture.layout, &segments, query, *row)?;
                    outcome.batches.clear();
                    samples[idx].push(outcome);
                }
            }

            let timings: Vec<Timing> = validated
                .iter()
                .zip(samples)
                .map(|(row, mut runs)| {
                    runs.sort_by_key(|run| run.wall);
                    let min = runs.first().map(|run| run.wall).unwrap_or_default();
                    let max = runs.last().map(|run| run.wall).unwrap_or_default();
                    let median = &runs[runs.len() / 2];
                    Timing {
                        label: row.label(),
                        median: median.wall,
                        min,
                        max,
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
    let plan = vortex_morsel_push::build_plan(
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
            "| {} | {} [{},{}] | {} | {} | {} | {} | {} | {} | {} | {} |",
            timing.label,
            millis(timing.median),
            millis(timing.min),
            millis(timing.max),
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

fn env_rows(name: &str) -> VortexResult<Option<usize>> {
    std::env::var(name)
        .ok()
        .map(|value| {
            value
                .parse::<usize>()
                .map_err(|err| vortex_error::vortex_err!("invalid {name} value `{value}`: {err}"))
        })
        .transpose()
}

fn parse_row_sizes(name: &str, value: &str) -> VortexResult<Vec<u64>> {
    let sizes: Vec<u64> = value
        .split(',')
        .map(str::trim)
        .map(|size| {
            size.parse::<u64>()
                .map_err(|err| vortex_error::vortex_err!("invalid {name} value `{size}`: {err}"))
        })
        .collect::<VortexResult<_>>()?;
    if sizes.is_empty() || sizes.contains(&0) {
        vortex_error::vortex_bail!("{name} must contain positive comma-separated row counts");
    }
    Ok(sizes)
}

/// Format a duration in milliseconds, avoiding `Debug` formatting.
fn millis(duration: Duration) -> String {
    format!("{:.3}ms", duration.as_secs_f64() * 1000.0)
}

fn opt(value: Option<u64>) -> String {
    value.map(|v| v.to_string()).unwrap_or_else(|| "—".into())
}
