// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Drives a prepared scan on the session's runtime.
//!
//! A planner says what it needs: compute, or reads. This driver decides where each runs. It
//! composes the planners over a [`PreparedScan`], the pruning plans first when the scan has a
//! filter and the chunks of the rows they keep after, and runs them on planning runs it owns:
//! the root run stepped on `spawn_cpu` tasks as the stream is polled, each spawned chunk on a
//! task of its own, up to the concurrency, or on the root run when driving chunks inline. Each
//! root reads through its own IO session. Per-batch output mapping runs on the task that
//! produced the batch, beside the scan's other chunks.

use std::future;
use std::sync::Arc;
use std::task::Poll;

use async_lock::Semaphore;
use async_lock::SemaphoreGuardArc;
use futures::SinkExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
use futures::future::Either;
use futures::stream;
use futures::stream::BoxStream;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_io::request::IoService;
use vortex_io::runtime::Handle;
use vortex_mask::Mask;

use crate::planning::driver::Progress;
use crate::planning::driver::Run;
use crate::planning::planner::Planner;
use crate::planning::planner::WorkScope;
use crate::prepare::ChunkPlanner;
use crate::prepare::PreparedScan;
use crate::prepare::PrunePlanner;
use crate::prepare::ScanBuilder;

/// Runs a scan on the session's runtime.
pub struct ScanDriver {
    handle: Handle,
    concurrency: usize,
    budget: Option<ChunkBudget>,
    ordered: bool,
}

/// A bound on the chunks in flight that the scans of several drivers share, so a scan with more
/// work left can use the slots the others leave idle.
#[derive(Clone)]
pub struct ChunkBudget {
    chunks: usize,
    slots: Arc<Semaphore>,
}

impl ChunkBudget {
    /// A budget of `chunks` chunks in flight at once.
    pub fn new(chunks: usize) -> Self {
        Self {
            chunks,
            slots: Arc::new(Semaphore::new(chunks)),
        }
    }
}

impl ScanDriver {
    /// A driver spawning on `handle`, with at most `concurrency` chunks in flight at once, whatever
    /// the machine's core count. If `concurrency` is zero, chunks run inline on the root run,
    /// which interleaves every chunk's reads. When `ordered`, the driver holds batches until the
    /// scan is done and yields them in row order.
    pub fn new(handle: Handle, concurrency: usize, ordered: bool) -> Self {
        Self {
            handle,
            concurrency,
            budget: None,
            ordered,
        }
    }

    /// A driver spawning on `handle` whose chunks take their slots from `budget`, shared with
    /// every other driver holding it. Each driver can also always run one chunk of its own, so a
    /// scan whose output is not being read cannot hold up the others.
    pub fn with_budget(handle: Handle, budget: ChunkBudget, ordered: bool) -> Self {
        Self {
            handle,
            concurrency: budget.chunks + 1,
            budget: Some(budget),
            ordered,
        }
    }

    /// Prepares `builder` where compute runs, composes its planners, and streams their batches.
    /// Preparation errors are the stream's first item.
    pub fn run(
        self,
        builder: impl Into<ScanBuilder>,
    ) -> BoxStream<'static, VortexResult<ArrayRef>> {
        self.run_mapped(builder, Ok)
    }

    /// Prepares and runs a scan, applying `map_fn` to each batch on the worker producing it.
    ///
    /// Output adapters can perform per-batch work here so it runs alongside the scan's other
    /// chunks instead of on the single task polling the output stream. Mapping errors are
    /// yielded through that stream.
    pub fn run_mapped<A: Send + 'static>(
        self,
        builder: impl Into<ScanBuilder>,
        map_fn: impl Fn(ArrayRef) -> VortexResult<A> + Send + Sync + 'static,
    ) -> BoxStream<'static, VortexResult<A>> {
        let builder = builder.into();
        let handle = self.handle.clone();
        stream::once(async move {
            handle
                .spawn_cpu(move || {
                    let scan = builder.prepare()?;
                    VortexResult::Ok(self.run_prepared(scan, map_fn))
                })
                .await
        })
        .try_flatten()
        .boxed()
    }

    /// Runs an already prepared scan, applying `map_fn` on the task that produced each batch.
    /// Use this when preparation errors must be reported before returning the stream.
    pub fn run_prepared<A: Send + 'static>(
        self,
        scan: PreparedScan,
        map_fn: impl Fn(ArrayRef) -> VortexResult<A> + Send + Sync + 'static,
    ) -> BoxStream<'static, VortexResult<A>> {
        let scan = Arc::new(scan);
        let map_fn: MapFn<A> = Arc::new(map_fn);
        let in_flight = self.concurrency;
        let inline = in_flight == 0;
        let scope = WorkScope {
            file_ordinal: 0,
            rows: scan.rows().clone(),
        };
        let io = scan.io();
        let root = compose(&scan);
        let (chunk_handle, chunk_io, chunk_map) =
            (self.handle.clone(), Arc::clone(&io), Arc::clone(&map_fn));
        let slots = self
            .budget
            .map(|budget| (Arc::new(Semaphore::new(1)), budget.slots));
        let batches = run_stream(self.handle, root, scope, io, map_fn, inline)
            .map_ok(move |yielded| match yielded {
                Yielded::Batch(start, batch) => {
                    stream::once(future::ready(Ok((start, batch)))).boxed()
                }
                Yielded::Spawn(scope, chunk) => {
                    let (handle, io, map_fn) = (
                        chunk_handle.clone(),
                        Arc::clone(&chunk_io),
                        Arc::clone(&chunk_map),
                    );
                    let Some((own, shared)) = slots.clone() else {
                        return chunk_stream(&handle, chunk, scope, io, map_fn);
                    };
                    stream::once(async move {
                        let slot = acquire_slot(own, shared).await;
                        chunk_stream(&handle, chunk, scope, io, map_fn).map(move |batch| {
                            let _ = &slot;
                            batch
                        })
                    })
                    .flatten()
                    .boxed()
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
            let mut held: Vec<(u64, A)> = batches.try_collect().await?;
            // Morsels finish in whatever order their reads arrive.
            held.sort_by_key(|(start, _)| *start);
            VortexResult::Ok(stream::iter(held.into_iter().map(|(_, batch)| Ok(batch))))
        })
        .try_flatten()
        .boxed()
    }
}

/// Waits for a slot to run a chunk in: the driver's own slot or one from the shared budget,
/// whichever frees first. The chunk holds it until it ends.
async fn acquire_slot(own: Arc<Semaphore>, shared: Arc<Semaphore>) -> SemaphoreGuardArc {
    if let Some(slot) = own.try_acquire_arc().or_else(|| shared.try_acquire_arc()) {
        return slot;
    }
    match futures::future::select(Box::pin(own.acquire_arc()), Box::pin(shared.acquire_arc())).await
    {
        Either::Left((slot, _)) | Either::Right((slot, _)) => slot,
    }
}

/// The planners of a scan, composed: with a filter, its pruning plans first over the whole scan,
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

/// A mapping applied to each batch on the task that produced it.
type MapFn<A> = Arc<dyn Fn(ArrayRef) -> VortexResult<A> + Send + Sync>;

/// What the root run's stream yields: a mapped batch with the row it starts at, or a chunk a
/// planner of the run spawned.
enum Yielded<A> {
    Batch(u64, A),
    Spawn(WorkScope, Box<dyn Planner>),
}

/// One CPU step of a run, taken on a `spawn_cpu` task with the run.
enum Step<A> {
    Batch(u64, A),
    Spawn(WorkScope, Box<dyn Planner>),
    RootDone,
    Idle,
    Waiting,
}

/// The state of the root run's stream.
struct RunState<A> {
    handle: Handle,
    /// Taken while a CPU step runs on its task and put back with its result.
    run: Option<Run>,
    /// Each root owns a session that the run clears when the root retires.
    io: Arc<dyn IoService>,
    map_fn: MapFn<A>,
    /// Whether a spawned chunk runs on this run, as another root, instead of being yielded.
    inline: bool,
}

/// Runs `root` over `scope` on a planning run of its own, reading through `io`, each CPU step
/// on a `spawn_cpu` task, and yields its mapped batches and, unless `inline`, the chunks it
/// spawns.
fn run_stream<A: Send + 'static>(
    handle: Handle,
    root: Box<dyn Planner>,
    scope: WorkScope,
    io: Arc<dyn IoService>,
    map_fn: MapFn<A>,
    inline: bool,
) -> BoxStream<'static, VortexResult<Yielded<A>>> {
    let mut run = Run::new();
    run.admit(root, scope, io.session());
    let state = RunState {
        handle,
        run: Some(run),
        io,
        map_fn,
        inline,
    };
    stream::try_unfold(state, |mut state| async move {
        loop {
            let run = state
                .run
                .take()
                .ok_or_else(|| vortex_err!("the scan's run is missing"))?;
            let map_fn = Arc::clone(&state.map_fn);
            let (run, step) = state
                .handle
                .spawn_cpu(move || {
                    let mut run = run;
                    let step = run.advance().and_then(|progress| {
                        Ok(match progress {
                            Progress::Batch(batch) => {
                                Step::Batch(batch.scope.rows.start, map_fn(batch.array)?)
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
                    run.admit(child, scope, state.io.session());
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
fn chunk_stream<A: Send + 'static>(
    handle: &Handle,
    root: Box<dyn Planner>,
    scope: WorkScope,
    io: Arc<dyn IoService>,
    map_fn: MapFn<A>,
) -> BoxStream<'static, VortexResult<(u64, A)>> {
    let (mut send, recv) = mpsc::channel(1);
    let task = handle.spawn(async move {
        let mut run = Run::new();
        run.admit(root, scope, io.session());
        loop {
            match run.advance()? {
                Progress::Batch(batch) => {
                    let start = batch.scope.rows.start;
                    let batch = map_fn(batch.array);
                    let failed = batch.is_err();
                    if send.send(batch.map(|batch| (start, batch))).await.is_err() || failed {
                        return Ok(());
                    }
                }
                Progress::Spawn(scope, child) => {
                    run.admit(child, scope, io.session());
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

#[cfg(test)]
mod tests;
