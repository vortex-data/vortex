// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Drives a V2 scan for DataFusion.
//!
//! A planner says what it needs: compute, or reads. This driver decides where each runs. The
//! opener composes the planners over a [`PreparedScan`] and hands the driver the first; the
//! driver runs it and everything it emits on planning runs it owns: the root run stepped on
//! `spawn_cpu` tasks as the stream is polled, each spawned chunk on a task of its own, or on the
//! root run when the engine's partitions already fill the cores. Reads use the prepared scan's
//! IO service, which may be a segment source or a completion-based reader. A worker-owned ring
//! keeps a partition's run on one worker. Arrow mapping may run on the consuming DataFusion task.

use std::future;
use std::sync::Arc;
use std::task::Poll;

use arrow_array::RecordBatch;
use futures::SinkExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
use futures::stream;
use futures::stream::BoxStream;
use tokio::sync::Semaphore;
use vortex::array::ArrayRef;
use vortex::array::arrays::Chunked;
use vortex::array::arrays::chunked::ChunkedArrayExt;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::io::request::IoService;
use vortex::io::runtime::Handle;
#[cfg(all(target_os = "linux", feature = "io-uring"))]
use vortex::io::uring::UringWorkerPool;
use vortex::layout::scan::v2::PreparedScan;
use vortex::scan::planning::driver::Progress;
use vortex::scan::planning::driver::Run;
use vortex::scan::planning::planner::Planner;
use vortex::scan::planning::planner::WorkScope;
use vortex_utils::parallelism::get_available_parallelism;

#[cfg(test)]
mod tests;

/// Maps a scan's batches to Arrow on the task that produced them.
type MapBatch = Arc<dyn Fn(ArrayRef) -> VortexResult<ScanBatch> + Send + Sync>;
type MapArrow = Arc<dyn Fn(ArrayRef) -> VortexResult<RecordBatch> + Send + Sync>;

enum ScanBatch {
    Array(ArrayRef),
    Arrow(RecordBatch),
}

/// Runs one partition's V2 scan on DataFusion's runtime.
pub struct ScanDriver {
    handle: Handle,
    /// Chunks run beside each other per worker thread; at 1 they run on the scan's own run.
    concurrency: usize,
    /// Whether batches are yielded in row order, which holds them until the scan is done.
    ordered: bool,
    consumer_mapping: bool,
    mapping_concurrency: usize,
    chunk_admission: Option<Arc<Semaphore>>,
    admit_after_io: bool,
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    workers: Option<UringWorkerPool>,
}

impl ScanDriver {
    pub(crate) fn new(handle: Handle, concurrency: usize, ordered: bool) -> Self {
        Self {
            handle,
            concurrency,
            ordered,
            consumer_mapping: false,
            mapping_concurrency: 1,
            chunk_admission: None,
            admit_after_io: false,
            #[cfg(all(target_os = "linux", feature = "io-uring"))]
            workers: None,
        }
    }

    /// Drives a partition's planning run on one persistent worker and its local ring.
    /// With control admission enabled, spawned chunks can start on other workers. Every
    /// completion scope is created on its assigned worker and stays there until it retires.
    #[cfg(all(target_os = "linux", feature = "io-uring"))]
    pub fn with_uring_workers(mut self, workers: UringWorkerPool) -> Self {
        self.workers = Some(workers);
        self
    }

    /// Limits active spawned chunks across drivers sharing this semaphore.
    ///
    /// A chunk holds its permit while computing or awaiting reads, and releases it before
    /// sending a batch to its consumer. Native worker pools provide their own admission and
    /// cannot be combined with this limit. With `after_io`, admission starts only after a
    /// chunk first waits for I/O; chunks served immediately from cache remain unrestricted.
    pub fn with_chunk_admission(mut self, admission: Arc<Semaphore>, after_io: bool) -> Self {
        self.chunk_admission = Some(admission);
        self.admit_after_io = after_io;
        self
    }

    pub(crate) fn with_consumer_mapping(mut self, enabled: bool, concurrency: usize) -> Self {
        self.consumer_mapping = enabled;
        self.mapping_concurrency = concurrency.max(1);
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
        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        if self.workers.is_some() && self.chunk_admission.is_some() {
            return stream::once(future::ready(Err(vortex_err!(
                "Native scan workers already control chunk admission"
            ))))
            .boxed();
        }
        let output_map = Arc::new(map);
        let output_handle = self.handle.clone();
        let map: MapBatch = if self.consumer_mapping {
            Arc::new(|array| Ok(ScanBatch::Array(array)))
        } else {
            let output_map = Arc::clone(&output_map);
            Arc::new(move |array| output_map(array).map(ScanBatch::Arrow))
        };
        let in_flight = self.concurrency * get_available_parallelism().unwrap_or(1);
        let inline = in_flight <= 1 && self.chunk_admission.is_none();
        let chunk_admission = self.chunk_admission;
        let scope = WorkScope {
            file_ordinal: 0,
            rows: scan.rows().clone(),
        };
        let io = scan.io();
        let (chunk_handle, chunk_io, chunk_map) =
            (self.handle.clone(), Arc::clone(&io), Arc::clone(&map));
        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        let chunk_workers = self.workers.clone();
        #[cfg(all(target_os = "linux", feature = "io-uring"))]
        let batches = if let Some(workers) = self.workers {
            // Without a filter, consumer mapping already spreads decoding over the engine's
            // tasks. Redispatching each chunk adds a worker handoff without moving that work.
            let dispatch_children = !inline
                && workers.max_control_jobs() > 0
                && (!self.consumer_mapping || scan.filter().is_some());
            worker_stream(
                workers,
                root,
                scope,
                Arc::clone(&io),
                map,
                dispatch_children,
            )
        } else {
            run_stream(self.handle, root, scope, io, map, inline)
        };
        #[cfg(not(all(target_os = "linux", feature = "io-uring")))]
        let batches = run_stream(self.handle, root, scope, io, map, inline);
        let batches = batches
            .map_ok(move |yielded| match yielded {
                Yielded::Batch(start, batch) => {
                    stream::once(future::ready(Ok((start, batch)))).boxed()
                }
                Yielded::Spawn(scope, chunk) => {
                    #[cfg(all(target_os = "linux", feature = "io-uring"))]
                    if let Some(workers) = &chunk_workers {
                        return worker_chunk_stream(
                            workers.clone(),
                            chunk,
                            scope,
                            Arc::clone(&chunk_io),
                            Arc::clone(&chunk_map),
                        );
                    }
                    chunk_stream(
                        &chunk_handle,
                        chunk,
                        scope,
                        Arc::clone(&chunk_io),
                        Arc::clone(&chunk_map),
                        chunk_admission.clone(),
                        self.admit_after_io,
                    )
                }
            })
            .try_flatten_unordered(in_flight);
        // The prepared scan holds the file's shared cache: keep it until the stream ends, so
        // scans of the file that start meanwhile find it.
        let batches = batches.chain(stream::poll_fn(move |_| {
            let _ = &scan;
            Poll::Ready(None)
        }));
        let batches = if self.ordered {
            stream::once(async move {
                let mut held: Vec<(u64, ScanBatch)> = batches.try_collect().await?;
                held.sort_by_key(|(start, _)| *start);
                VortexResult::Ok(stream::iter(held.into_iter().map(|(_, batch)| Ok(batch))))
            })
            .try_flatten()
            .boxed()
        } else {
            batches.map_ok(|(_, batch)| batch).boxed()
        };
        if !self.consumer_mapping || self.mapping_concurrency == 1 {
            return batches
                .and_then(move |batch| {
                    future::ready(match batch {
                        ScanBatch::Array(array) => output_map(array),
                        ScanBatch::Arrow(batch) => Ok(batch),
                    })
                })
                .boxed();
        }
        map_chunks(
            batches,
            output_handle,
            output_map,
            self.ordered,
            in_flight.min(self.mapping_concurrency),
        )
    }
}

fn map_chunks(
    batches: BoxStream<'static, VortexResult<ScanBatch>>,
    handle: Handle,
    map: MapArrow,
    ordered: bool,
    in_flight: usize,
) -> BoxStream<'static, VortexResult<RecordBatch>> {
    let batches = batches
        .map_ok(|batch| {
            // Stored chunks can contain expensive independent decoders. Export them
            // separately instead of merging their fields and decoding them serially.
            let chunks = match batch {
                ScanBatch::Array(array) => match array.as_opt::<Chunked>() {
                    Some(chunks) => chunks
                        .iter_chunks()
                        .cloned()
                        .map(ScanBatch::Array)
                        .collect(),
                    None => vec![ScanBatch::Array(array)],
                },
                ScanBatch::Arrow(batch) => vec![ScanBatch::Arrow(batch)],
            };
            stream::iter(chunks.into_iter().map(Ok))
        })
        .try_flatten()
        .map_ok(move |batch| {
            let map = Arc::clone(&map);
            handle.spawn_cpu(move || match batch {
                ScanBatch::Array(array) => map(array),
                ScanBatch::Arrow(batch) => Ok(batch),
            })
        });
    if ordered {
        batches.try_buffered(in_flight.max(1)).boxed()
    } else {
        batches.try_buffer_unordered(in_flight.max(1)).boxed()
    }
}

/// What the root run's stream yields: a mapped batch with the row it starts at, or a chunk a
/// planner of the run spawned.
enum Yielded {
    Batch(u64, ScanBatch),
    Spawn(WorkScope, Box<dyn Planner>),
}

/// One CPU step of a run, taken on a `spawn_cpu` task with the run.
enum Step {
    Batch(u64, ScanBatch),
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
    /// Each admitted root needs its own session because retiring a root clears its session.
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
    io: Arc<dyn IoService>,
    map: MapBatch,
    admission: Option<Arc<Semaphore>>,
    admit_after_io: bool,
) -> BoxStream<'static, VortexResult<(u64, ScanBatch)>> {
    let (mut send, recv) = mpsc::channel(1);
    let task = handle.spawn(async move {
        let mut run = Run::new();
        run.admit(root, scope, io.session());
        let mut permit = None;
        let mut needs_admission = !admit_after_io;
        loop {
            if needs_admission
                && permit.is_none()
                && let Some(admission) = &admission
            {
                permit = Some(
                    Arc::clone(admission)
                        .acquire_owned()
                        .await
                        .map_err(|_| vortex_err!("Scan chunk admission closed"))?,
                );
            }
            match run.advance()? {
                Progress::Batch(batch) => {
                    let start = batch.scope.rows.start;
                    let batch = map(batch.array);
                    let failed = batch.is_err();
                    // A slow consumer must not retain admission needed by another scan.
                    drop(permit.take());
                    if send.send(batch.map(|batch| (start, batch))).await.is_err() || failed {
                        return Ok(());
                    }
                }
                Progress::Spawn(scope, child) => {
                    run.admit(child, scope, io.session());
                }
                Progress::RootDone(_) if run.live_roots() > 0 => {}
                Progress::RootDone(_) | Progress::Idle => return Ok(()),
                Progress::Waiting if !needs_admission => needs_admission = true,
                Progress::Waiting => future::poll_fn(|cx| run.poll_completion(cx)).await?,
            }
        }
    });
    let errors = stream::once(task)
        .filter_map(|result: VortexResult<()>| future::ready(result.err().map(Err)));
    stream::select(recv, errors).boxed()
}

/// A native completion scope is created and driven on the same worker for its whole lifetime.
#[cfg(all(target_os = "linux", feature = "io-uring"))]
fn worker_stream(
    workers: UringWorkerPool,
    root: Box<dyn Planner>,
    scope: WorkScope,
    io: Arc<dyn IoService>,
    map: MapBatch,
    dispatch_children: bool,
) -> BoxStream<'static, VortexResult<Yielded>> {
    let (mut send, recv) = mpsc::channel(1);
    let work = move || async move {
        let mut run = Run::new();
        run.admit(root, scope, io.session());
        loop {
            match run.advance()? {
                Progress::Batch(batch) => {
                    let result = map(batch.array)
                        .map(|batch_array| Yielded::Batch(batch.scope.rows.start, batch_array));
                    let failed = result.is_err();
                    if send.send(result).await.is_err() || failed {
                        return Ok(());
                    }
                }
                Progress::Spawn(scope, child) if dispatch_children => {
                    if send.send(Ok(Yielded::Spawn(scope, child))).await.is_err() {
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
    };
    let task = if dispatch_children {
        workers.spawn_control(work)
    } else {
        workers.spawn_async(work)
    };
    let errors = stream::once(task)
        .filter_map(|result: VortexResult<()>| future::ready(result.err().map(Err)));
    // Retain the pool while its task and the task's output stream are alive.
    stream::select(recv, errors)
        .chain(stream::poll_fn(move |_| {
            let _ = &workers;
            Poll::Ready(None)
        }))
        .boxed()
}

#[cfg(all(target_os = "linux", feature = "io-uring"))]
fn worker_chunk_stream(
    workers: UringWorkerPool,
    root: Box<dyn Planner>,
    scope: WorkScope,
    io: Arc<dyn IoService>,
    map: MapBatch,
) -> BoxStream<'static, VortexResult<(u64, ScanBatch)>> {
    worker_stream(workers, root, scope, io, map, false)
        .and_then(|yielded| {
            future::ready(match yielded {
                Yielded::Batch(start, batch) => Ok((start, batch)),
                Yielded::Spawn(..) => Err(vortex_err!(
                    "A native chunk must admit its child planners inline"
                )),
            })
        })
        .boxed()
}
