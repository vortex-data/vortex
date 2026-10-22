// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Drives a V2 scan of a file for callers without an engine of their own.
//!
//! A planner says what it needs: compute, or reads. This driver decides where each runs. It
//! composes the planners over a [`PreparedScan`], the zone proofs first when the scan has a
//! filter and the chunks of the rows they keep after, and runs them on planning runs it owns:
//! the root run stepped on `spawn_cpu` tasks as the stream is polled, each spawned chunk on a
//! task of its own, up to the concurrency, or on the root run at a concurrency of one. Every
//! read is a `spawn_io` task of the file's segment source.

use std::future;
use std::sync::Arc;
use std::task::Poll;

use futures::SinkExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::request::IoSource;
use vortex_io::runtime::Handle;
use vortex_layout::scan::v2::ChunkPlanner;
use vortex_layout::scan::v2::PreparedScan;
use vortex_layout::scan::v2::PrunePlanner;
use vortex_layout::scan::v2::ScanBuilder;
use vortex_mask::Mask;
use vortex_scan::planning::driver::Progress;
use vortex_scan::planning::driver::Run;
use vortex_scan::planning::planner::Planner;
use vortex_scan::planning::planner::WorkScope;
use vortex_utils::parallelism::get_available_parallelism;

/// Runs a V2 scan on the session's runtime.
pub struct ScanDriver {
    handle: Handle,
    concurrency: usize,
    ordered: bool,
}

impl ScanDriver {
    /// A driver spawning on `handle`, running `concurrency` chunks beside each other per worker
    /// thread, or every chunk on the scan's own run at one, and yielding batches in row order
    /// when `ordered`, which holds them until the scan is done.
    pub fn new(handle: Handle, concurrency: usize, ordered: bool) -> Self {
        assert!(concurrency > 0);
        Self {
            handle,
            concurrency,
            ordered,
        }
    }

    /// Prepares `builder` where compute runs, composes its planners, and streams their batches.
    /// Preparation errors are the stream's first item.
    pub fn run(self, builder: ScanBuilder) -> BoxStream<'static, VortexResult<ArrayRef>> {
        let handle = self.handle.clone();
        stream::once(async move {
            handle
                .spawn_cpu(move || {
                    let scan = Arc::new(builder.prepare()?);
                    VortexResult::Ok(self.execute(scan))
                })
                .await
        })
        .try_flatten()
        .boxed()
    }

    fn execute(self, scan: Arc<PreparedScan>) -> BoxStream<'static, VortexResult<ArrayRef>> {
        let in_flight = self.concurrency * get_available_parallelism().unwrap_or(1);
        let inline = in_flight <= 1;
        let scope = WorkScope {
            file_ordinal: 0,
            rows: scan.rows().clone(),
        };
        let io = scan.io();
        let root = compose(&scan);
        let (chunk_handle, chunk_io) = (self.handle.clone(), Arc::clone(&io));
        let batches = run_stream(self.handle, root, scope, io.session(), inline)
            .map_ok(move |yielded| match yielded {
                Yielded::Batch(start, batch) => {
                    stream::once(future::ready(Ok((start, batch)))).boxed()
                }
                Yielded::Spawn(scope, chunk) => {
                    chunk_stream(&chunk_handle, chunk, scope, chunk_io.session())
                }
            })
            .try_flatten_unordered(in_flight);
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
            let mut held: Vec<(u64, ArrayRef)> = batches.try_collect().await?;
            // Morsels finish in whatever order their reads arrive.
            held.sort_by_key(|(start, _)| *start);
            VortexResult::Ok(stream::iter(held.into_iter().map(|(_, batch)| Ok(batch))))
        })
        .try_flatten()
        .boxed()
    }
}

/// The planners of a scan, composed: with a filter, its proofs first over the whole scan,
/// continuing into the chunks of the rows they keep; without, the chunks directly.
fn compose(scan: &Arc<PreparedScan>) -> Box<dyn Planner> {
    let rows = scan.rows().clone();
    let chunks = {
        let scan = Arc::clone(scan);
        let rows = rows.clone();
        move |kept: Option<Mask>| -> Box<dyn Planner> {
            let selection = scan.selection().clone();
            Box::new(ChunkPlanner::new(scan, rows, selection, kept))
        }
    };
    if scan.filter().is_some() {
        Box::new(PrunePlanner::new(Arc::clone(scan), rows, Box::new(chunks)))
    } else {
        chunks(None)
    }
}

/// What the root run's stream yields: a batch with the row it starts at, or a chunk a planner of
/// the run spawned.
enum Yielded {
    Batch(u64, ArrayRef),
    Spawn(WorkScope, Box<dyn Planner>),
}

/// One CPU step of a run, taken on a `spawn_cpu` task with the run.
enum Step {
    Batch(u64, ArrayRef),
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
    /// Whether a spawned chunk runs on this run, as another root, instead of being yielded.
    inline: bool,
}

/// Runs `root` over `scope` on a planning run of its own, reading through `io`, each CPU step
/// on a `spawn_cpu` task, and yields its batches and, unless `inline`, the chunks it spawns.
fn run_stream(
    handle: Handle,
    root: Box<dyn Planner>,
    scope: WorkScope,
    io: Arc<dyn IoSource>,
    inline: bool,
) -> BoxStream<'static, VortexResult<Yielded>> {
    let mut run = Run::new();
    run.admit(root, scope, Arc::clone(&io));
    let state = RunState {
        handle,
        run: Some(run),
        io,
        inline,
    };
    stream::try_unfold(state, |mut state| async move {
        loop {
            let run = state
                .run
                .take()
                .ok_or_else(|| vortex_err!("the scan's run is missing"))?;
            let (run, step) = state
                .handle
                .spawn_cpu(move || {
                    let mut run = run;
                    let step = run.advance().map(|progress| match progress {
                        Progress::Batch(batch) => Step::Batch(batch.scope.rows.start, batch.array),
                        Progress::Spawn(scope, child) => Step::Spawn(scope, child),
                        Progress::RootDone(_) => Step::RootDone,
                        Progress::Idle => Step::Idle,
                        Progress::Waiting => Step::Waiting,
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
/// drives the run to its end and awaits only its reads, and yields its batches with the row each
/// starts at. A planner the chunk spawns in turn runs as another root of the same run.
fn chunk_stream(
    handle: &Handle,
    root: Box<dyn Planner>,
    scope: WorkScope,
    io: Arc<dyn IoSource>,
) -> BoxStream<'static, VortexResult<(u64, ArrayRef)>> {
    let (mut send, recv) = mpsc::channel(1);
    let task = handle.spawn(async move {
        let mut run = Run::new();
        run.admit(root, scope, Arc::clone(&io));
        loop {
            match run.advance()? {
                Progress::Batch(batch) => {
                    if send
                        .send(Ok((batch.scope.rows.start, batch.array)))
                        .await
                        .is_err()
                    {
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
