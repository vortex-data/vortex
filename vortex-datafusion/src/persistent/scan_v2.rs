// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Drives a V2 scan for DataFusion.
//!
//! A planner says what it needs: compute, or reads. This driver decides where each runs. The
//! opener composes the planners over a [`PreparedScan`] and hands the driver the first; the
//! driver runs it and everything it emits on planning runs it owns: the root run stepped on
//! `spawn_cpu` tasks as the stream is polled, each spawned chunk on a task of its own, or on the
//! root run when the engine's partitions already fill the cores. Every read is a `spawn_io` task
//! of the file's segment source. The DataFusion thread polling the stream does no scan work
//! itself.

use std::future;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Poll;

use arrow_array::RecordBatch;
use futures::SinkExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
use futures::stream;
use futures::stream::BoxStream;
use vortex::array::ArrayRef;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::io::request::IoSource;
use vortex::io::runtime::Handle;
use vortex::layout::scan::v2::ChunkPlanner;
use vortex::layout::scan::v2::PreparedScan;
use vortex::layout::scan::v2::PrunePlanner;
use vortex::mask::Mask;
use vortex::scan::planning::driver::Progress;
use vortex::scan::planning::driver::Run;
use vortex::scan::planning::planner::Planner;
use vortex::scan::planning::planner::WorkScope;
use vortex_utils::parallelism::get_available_parallelism;

#[cfg(test)]
mod tests;

/// Maps a scan's batches to Arrow on the task that produced them.
type MapBatch = Arc<dyn Fn(ArrayRef) -> VortexResult<RecordBatch> + Send + Sync>;

const INITIAL_CHUNK_ROWS: u64 = 8192;

/// Claims a partition's startup budget across its files, without duplicating their rows.
pub(crate) fn initial_ranges(rows: Range<u64>, remaining: &AtomicUsize) -> Vec<Range<u64>> {
    let needed =
        usize::try_from((rows.end - rows.start).div_ceil(INITIAL_CHUNK_ROWS)).unwrap_or(usize::MAX);
    let mut previous = remaining.load(Ordering::Relaxed);
    let claimed = loop {
        let claimed = previous.min(needed);
        if claimed == 0 {
            break 0;
        }
        match remaining.compare_exchange_weak(
            previous,
            previous - claimed,
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            Ok(_) => break claimed,
            Err(current) => previous = current,
        }
    };
    (0..claimed)
        .map(|index| {
            let start = rows.start + index as u64 * INITIAL_CHUNK_ROWS;
            start..start.saturating_add(INITIAL_CHUNK_ROWS).min(rows.end)
        })
        .collect()
}

/// Composes pruning, filtering and projection for an independent file-local range.
pub(crate) fn scan_planner(scan: Arc<PreparedScan>, rows: Range<u64>) -> Box<dyn Planner> {
    let chunks = {
        let scan = Arc::clone(&scan);
        let rows = rows.clone();
        move |kept: Option<Mask>| -> Box<dyn Planner> {
            let selection = scan.selection().clone();
            Box::new(ChunkPlanner::new(scan, rows, selection, kept))
        }
    };
    if scan.filter().is_some() {
        Box::new(PrunePlanner::new(scan, rows, Box::new(chunks)))
    } else {
        chunks(None)
    }
}

/// Runs one partition's V2 scan on DataFusion's runtime.
pub(crate) struct ScanDriver {
    handle: Handle,
    /// Chunks run beside each other per worker thread; at 1 they run on the scan's own run.
    concurrency: usize,
    /// Whether batches are yielded in row order, which holds them until the scan is done.
    ordered: bool,
}

impl ScanDriver {
    pub(crate) fn new(handle: Handle, concurrency: usize, ordered: bool) -> Self {
        Self {
            handle,
            concurrency,
            ordered,
        }
    }

    /// Runs `root`, the first planner composed over `scan`, and everything it emits, and
    /// streams their batches mapped by `map`.
    pub(crate) fn run(
        self,
        scan: Arc<PreparedScan>,
        root: Box<dyn Planner>,
        initial: Vec<Range<u64>>,
        map: impl Fn(ArrayRef) -> VortexResult<RecordBatch> + Send + Sync + 'static,
    ) -> BoxStream<'static, VortexResult<RecordBatch>> {
        let map: MapBatch = Arc::new(map);
        let in_flight = self.concurrency * get_available_parallelism().unwrap_or(1);
        let inline = in_flight <= 1;
        let scope = WorkScope {
            file_ordinal: 0,
            rows: scan.rows().clone(),
        };
        let io = scan.io();
        let initial_count = initial.len().max(1);
        let (initial_handle, initial_io, initial_map, initial_scan) = (
            self.handle.clone(),
            Arc::clone(&io),
            Arc::clone(&map),
            Arc::clone(&scan),
        );
        let initial = stream::iter(initial)
            .map(move |rows| {
                let scope = WorkScope {
                    file_ordinal: 0,
                    rows: rows.clone(),
                };
                chunk_stream(
                    &initial_handle,
                    scan_planner(Arc::clone(&initial_scan), rows),
                    scope,
                    initial_io.session(),
                    Arc::clone(&initial_map),
                )
            })
            .flatten_unordered(initial_count);
        let (chunk_handle, chunk_io, chunk_map) =
            (self.handle.clone(), Arc::clone(&io), Arc::clone(&map));
        let batches = run_stream(self.handle, root, scope, io.session(), map, inline)
            .map_ok(move |yielded| match yielded {
                Yielded::Batch(start, batch) => {
                    stream::once(future::ready(Ok((start, batch)))).boxed()
                }
                Yielded::Spawn(scope, chunk) => chunk_stream(
                    &chunk_handle,
                    chunk,
                    scope,
                    chunk_io.session(),
                    Arc::clone(&chunk_map),
                ),
            })
            .try_flatten_unordered(in_flight);
        // Drain startup batches before admitting bulk work, so TopK can tighten its bound.
        let batches = initial.chain(batches);
        // The prepared scan holds the file's shared cache: keep it until the stream ends, so
        // scans of the file that start meanwhile find it.
        let batches = batches.chain(stream::poll_fn(move |_| {
            let _ = &scan;
            Poll::Ready(None)
        }));
        if !self.ordered {
            return batches.map_ok(|(_, batch)| batch).boxed();
        }
        stream::once(async move {
            let mut held: Vec<(u64, RecordBatch)> = batches.try_collect().await?;
            // Morsels finish in whatever order their reads arrive.
            held.sort_by_key(|(start, _)| *start);
            VortexResult::Ok(stream::iter(held.into_iter().map(|(_, batch)| Ok(batch))))
        })
        .try_flatten()
        .boxed()
    }
}

/// What the root run's stream yields: a mapped batch with the row it starts at, or a chunk a
/// planner of the run spawned.
enum Yielded {
    Batch(u64, RecordBatch),
    Spawn(WorkScope, Box<dyn Planner>),
}

/// One CPU step of a run, taken on a `spawn_cpu` task with the run.
enum Step {
    Batch(u64, RecordBatch),
    Spawn(WorkScope, Box<dyn Planner>),
    RootDone,
    Idle,
    Waiting,
}

/// The state of the root run's stream.
struct RunState {
    handle: Handle,
    /// Taken while a CPU step runs on its task and put back with its result.
    run: Option<Run>,
    io: Arc<dyn IoSource>,
    map: MapBatch,
    /// Whether a spawned chunk runs on this run, as another root, instead of being yielded.
    inline: bool,
}

/// Runs `root` over `scope` on a planning run of its own, reading through `io`, each CPU step
/// on a `spawn_cpu` task, and yields its mapped batches and, unless `inline`, the chunks it
/// spawns. Inline, a spawned chunk runs on this run beside the others: the run's steps already
/// interleave every chunk's work, and a run per chunk would only add a hand-off per chunk for
/// parallelism the engine provides.
fn run_stream(
    handle: Handle,
    root: Box<dyn Planner>,
    scope: WorkScope,
    io: Arc<dyn IoSource>,
    map: MapBatch,
    inline: bool,
) -> BoxStream<'static, VortexResult<Yielded>> {
    let mut run = Run::new();
    run.admit(root, scope, Arc::clone(&io));
    let state = RunState {
        handle,
        run: Some(run),
        io,
        map,
        inline,
    };
    stream::try_unfold(state, |mut state| async move {
        loop {
            let run = state
                .run
                .take()
                .ok_or_else(|| vortex_err!("the scan's run is missing"))?;
            let map = Arc::clone(&state.map);
            let (run, step) = state
                .handle
                .spawn_cpu(move || {
                    let mut run = run;
                    let step = run.advance().and_then(|progress| {
                        Ok(match progress {
                            Progress::Batch(batch) => {
                                Step::Batch(batch.scope.rows.start, map(batch.array)?)
                            }
                            Progress::Spawn(scope, child) => Step::Spawn(scope, child),
                            Progress::RootDone(_) => Step::RootDone,
                            Progress::Idle => Step::Idle,
                            Progress::Waiting => Step::Waiting,
                        })
                    });
                    (run, step)
                })
                .await;
            let run = state.run.insert(run);
            match step? {
                Step::Batch(start, batch) => {
                    return Ok(Some((Yielded::Batch(start, batch), state)));
                }
                Step::Spawn(scope, child) if state.inline => {
                    run.admit(child, scope, Arc::clone(&state.io));
                }
                Step::Spawn(scope, child) => {
                    return Ok(Some((Yielded::Spawn(scope, child), state)));
                }
                Step::RootDone if run.live_roots() > 0 => {}
                Step::RootDone | Step::Idle => return Ok(None),
                Step::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
            }
        }
    })
    .boxed()
}

/// Runs a spawned chunk's `root` over `scope` on a planning run of its own, on one task that
/// drives the run to its end and awaits only its reads, and yields its mapped batches with the
/// row each starts at. A planner the chunk spawns in turn runs as another root of the same run.
///
/// One task per chunk, not one per step: a chunk's run waits for reads many times, and a task
/// hop at each wait is latency on the chunk's critical path that no other chunk's work fills.
fn chunk_stream(
    handle: &Handle,
    root: Box<dyn Planner>,
    scope: WorkScope,
    io: Arc<dyn IoSource>,
    map: MapBatch,
) -> BoxStream<'static, VortexResult<(u64, RecordBatch)>> {
    let (mut send, recv) = mpsc::channel(1);
    let task = handle.spawn(async move {
        let mut run = Run::new();
        run.admit(root, scope, Arc::clone(&io));
        loop {
            match run.advance()? {
                Progress::Batch(batch) => {
                    let start = batch.scope.rows.start;
                    let batch = map(batch.array);
                    let failed = batch.is_err();
                    if send.send(batch.map(|batch| (start, batch))).await.is_err() || failed {
                        return Ok(());
                    }
                }
                Progress::Spawn(scope, child) => {
                    run.admit(child, scope, Arc::clone(&io));
                }
                Progress::RootDone(_) if run.live_roots() > 0 => {}
                Progress::RootDone(_) | Progress::Idle => return Ok(()),
                Progress::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
            }
        }
    });
    let errors = stream::once(task)
        .filter_map(|result: VortexResult<()>| future::ready(result.err().map(Err)));
    stream::select(recv, errors).boxed()
}
