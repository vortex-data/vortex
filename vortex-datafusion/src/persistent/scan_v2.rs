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
use std::pin::Pin;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use arrow_array::RecordBatch;
use futures::FutureExt;
use futures::SinkExt;
use futures::Stream;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
use futures::future::BoxFuture;
use futures::stream;
use futures::stream::BoxStream;
use futures::stream::Fuse;
use futures::stream::FuturesOrdered;
use vortex::array::ArrayRef;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::io::request::IoService;
use vortex::io::request::IoSource;
use vortex::io::runtime::Handle;
use vortex::layout::scan::v2::PreparedScan;
use vortex::scan::planning::driver::Progress;
use vortex::scan::planning::driver::Run;
use vortex::scan::planning::planner::Planner;
use vortex::scan::planning::planner::WorkScope;
use vortex_utils::parallelism::get_available_parallelism;

/// Maps a scan's batches to Arrow on the task that produced them.
type MapBatch = Arc<dyn Fn(ArrayRef) -> VortexResult<RecordBatch> + Send + Sync>;

/// Runs one partition's V2 scan on DataFusion's runtime.
pub(crate) struct ScanDriver {
    handle: Handle,
    /// Chunks run beside each other per worker thread; at 1 they run on the scan's own run.
    concurrency: usize,
    /// Whether batches are yielded in row order, which holds each chunk's until it is done.
    ordered: bool,
    /// Whether the consumer stops after some rows, so the first chunks may be all it needs.
    limited: bool,
}

impl ScanDriver {
    pub(crate) fn new(handle: Handle, concurrency: usize, ordered: bool) -> Self {
        Self {
            handle,
            concurrency,
            ordered,
            limited: false,
        }
    }

    /// Marks the scan as read by a consumer that stops after some rows, such as a limit. An
    /// ordered scan then starts one chunk and runs more beside it only as chunks finish without
    /// the consumer stopping, doubling up to the driver's concurrency.
    pub(crate) fn with_limit(mut self, limited: bool) -> Self {
        self.limited = limited;
        self
    }

    /// Runs `root`, the first planner composed over `scan`, and everything it emits, and
    /// streams their batches mapped by `map`.
    pub(crate) fn run(
        self,
        scan: Arc<PreparedScan>,
        root: Box<dyn Planner>,
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
        let (chunk_handle, chunk_io, chunk_map) =
            (self.handle.clone(), Arc::clone(&io), Arc::clone(&map));
        let yielded = run_stream(self.handle, root, scope, Arc::clone(&io), map, inline);
        let chunk = move |scope: WorkScope, chunk: Box<dyn Planner>| {
            chunk_stream(
                &chunk_handle,
                chunk,
                scope,
                chunk_io.session(),
                Arc::clone(&chunk_map),
            )
        };
        let batches = if !self.ordered || inline {
            let batches = yielded
                .map_ok(move |yielded| match yielded {
                    Yielded::Batch(start, batch) => {
                        stream::once(future::ready(Ok((start, batch)))).boxed()
                    }
                    Yielded::Spawn(scope, planner) => chunk(scope, planner),
                })
                .try_flatten_unordered(in_flight);
            if !self.ordered {
                batches.map_ok(|(_, batch)| batch).boxed()
            } else {
                // Inline, every chunk runs on the root run, whose morsels finish in whatever
                // order their reads arrive: only the whole scan can be put in row order.
                stream::once(async move {
                    let mut held: Vec<(u64, RecordBatch)> = batches.try_collect().await?;
                    held.sort_by_key(|(start, _)| *start);
                    VortexResult::Ok(stream::iter(held.into_iter().map(|(_, batch)| Ok(batch))))
                })
                .try_flatten()
                .boxed()
            }
        } else {
            // The root run spawns its chunks in row order, over rows that do not overlap, so
            // putting each chunk's batches in order, and the chunks in the order they were
            // spawned, puts the scan in order. Up to `in_flight` chunks run ahead of the one
            // being yielded, and a consumer that stops early, such as a limit, stops the scan.
            let first_window = if self.limited { 1 } else { in_flight };
            let chunks = yielded
                .map_ok(move |yielded| match yielded {
                    Yielded::Batch(start, batch) => future::ready(Ok(vec![(start, batch)])).boxed(),
                    Yielded::Spawn(scope, planner) => {
                        let batches = chunk(scope, planner);
                        async move {
                            // Morsels finish in whatever order their reads arrive.
                            let mut held: Vec<(u64, RecordBatch)> = batches.try_collect().await?;
                            held.sort_by_key(|(start, _)| *start);
                            Ok(held)
                        }
                        .boxed()
                    }
                })
                .boxed();
            OrderedWindow::new(chunks, first_window, in_flight)
                .map_ok(|held| stream::iter(held.into_iter().map(|(_, batch)| Ok(batch))))
                .try_flatten()
                .boxed()
        };
        // The prepared scan holds the file's shared cache: keep it until the stream ends, so
        // scans of the file that start meanwhile find it.
        batches
            .chain(stream::poll_fn(move |_| {
                let _ = &scan;
                Poll::Ready(None)
            }))
            .boxed()
    }
}

/// A chunk's batches, put in row order once the chunk is done.
type HeldChunk = BoxFuture<'static, VortexResult<Vec<(u64, RecordBatch)>>>;

/// Runs the chunks of `source` beside each other and yields their results in the order they were
/// spawned, like [`TryStreamExt::try_buffered`], with a window that starts at `window` and
/// doubles each time a chunk is yielded, up to `max`. A consumer that needs only the first rows
/// stops the scan before chunks it never reads have started.
struct OrderedWindow {
    source: Fuse<BoxStream<'static, VortexResult<HeldChunk>>>,
    running: FuturesOrdered<HeldChunk>,
    window: usize,
    max: usize,
}

impl OrderedWindow {
    fn new(source: BoxStream<'static, VortexResult<HeldChunk>>, window: usize, max: usize) -> Self {
        Self {
            source: source.fuse(),
            running: FuturesOrdered::new(),
            window: window.clamp(1, max.max(1)),
            max: max.max(1),
        }
    }
}

impl Stream for OrderedWindow {
    type Item = VortexResult<Vec<(u64, RecordBatch)>>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        while this.running.len() < this.window {
            match this.source.poll_next_unpin(cx) {
                Poll::Ready(Some(Ok(chunk))) => this.running.push_back(chunk),
                Poll::Ready(Some(Err(error))) => return Poll::Ready(Some(Err(error))),
                Poll::Ready(None) | Poll::Pending => break,
            }
        }
        match this.running.poll_next_unpin(cx) {
            Poll::Ready(Some(held)) => {
                this.window = (this.window * 2).min(this.max);
                Poll::Ready(Some(held))
            }
            Poll::Ready(None) if this.source.is_done() => Poll::Ready(None),
            Poll::Ready(None) | Poll::Pending => Poll::Pending,
        }
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
    /// The scan's reads: each root of the run reads through a session of its own, which the run
    /// clears when the root retires.
    io: Arc<dyn IoService>,
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
    io: Arc<dyn IoService>,
    map: MapBatch,
    inline: bool,
) -> BoxStream<'static, VortexResult<Yielded>> {
    let mut run = Run::new();
    run.admit(root, scope, io.session());
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
