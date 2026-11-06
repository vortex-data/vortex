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

use std::collections::VecDeque;
use std::future;
use std::pin::pin;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;
use std::task::Poll;

use arrow_array::RecordBatch;
use datafusion_common::DataFusionError;
use datafusion_common::Result as DFResult;
use futures::SinkExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
use futures::stream;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use tokio::sync::Notify;
use vortex::array::ArrayRef;
use vortex::error::VortexError;
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
        let batches = run_stream(self.handle, root, scope, Arc::clone(&io), map, inline)
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

/// Maps a pooled chunk's batch to the scan's output, with the residual filter and the projection
/// of the file the chunk belongs to. `None` means the batch kept no rows.
pub(crate) type FinishBatch = Arc<dyn Fn(ArrayRef) -> DFResult<Option<RecordBatch>> + Send + Sync>;

/// The chunks of the files a scan's partitions have open, which any partition of the scan may run.
///
/// DataFusion assigns each partition its own byte ranges of the files, and shares only whole
/// ranges between partitions. A filter that keeps the rows of a few files leaves the partitions of
/// those files with all of the work, and the others idle. So a partition puts the chunks of the
/// range it opens here, and each partition takes chunks from here until none is left: the
/// partitions of the scan share the work of every open file, and each runs the operators above the
/// scan, such as a partial aggregate, on the batches of the chunks it takes.
///
/// A partition stream ends only when the pool has no chunk left that it may take and no root that
/// may still add one is running, so a chunk is never left without a stream to run it, and a
/// partition that finds the pool empty while another file is still being cut into chunks stays to
/// help with it.
#[derive(Default)]
pub(crate) struct ChunkPool {
    next_owner: AtomicU64,
    chunks: Mutex<VecDeque<PooledChunk>>,
    /// The roots still running, each of which may add chunks.
    roots: AtomicUsize,
    /// Notified when a chunk is added or a root finishes.
    changed: Notify,
}

/// A chunk of a file, with what it needs to run in any partition.
struct PooledChunk {
    /// The stream that opened the chunk's file.
    owner: u64,
    scope: WorkScope,
    planner: Box<dyn Planner>,
    file: Arc<PooledFile>,
}

/// What every chunk of one open file shares.
struct PooledFile {
    io: Arc<dyn IoService>,
    finish: FinishBatch,
    /// Holds the file's shared cache while any of its chunks can still run.
    _scan: Arc<PreparedScan>,
}

impl ChunkPool {
    /// Runs `root`, the first planner composed over `scan`, and puts the chunks it spawns in the
    /// pool. The stream yields the batches of the chunks it takes from the pool, finished by
    /// `finish` for this file's chunks, and by the finish of their own file for the others.
    ///
    /// With `help`, the stream takes any chunk, until the pool is empty. Without, it takes only the
    /// chunks of its own file, for a stream that may be dropped before it ends: a chunk it took
    /// from another file would lose its rows.
    pub(crate) fn run(
        self: &Arc<Self>,
        scan: Arc<PreparedScan>,
        root: Box<dyn Planner>,
        finish: FinishBatch,
        help: bool,
    ) -> BoxStream<'static, DFResult<RecordBatch>> {
        let owner = self.next_owner.fetch_add(1, Ordering::Relaxed);
        let io = scan.io();
        self.roots.fetch_add(1, Ordering::AcqRel);

        let mut run = Run::new();
        run.admit(
            root,
            WorkScope {
                file_ordinal: 0,
                rows: scan.rows().clone(),
            },
            io.session(),
        );

        let state = PoolStream {
            pool: Arc::clone(self),
            owner,
            help,
            root: Some(run),
            file: Arc::new(PooledFile {
                io,
                finish,
                _scan: scan,
            }),
            current: None,
        };
        stream::try_unfold(state, |mut state| async move {
            let batch = state.next_batch().await?;
            Ok(batch.map(|batch| (batch, state)))
        })
        .boxed()
    }

    /// Takes the oldest chunk the stream of `owner` may run.
    fn take(&self, owner: u64, help: bool) -> Option<PooledChunk> {
        let mut chunks = self.chunks.lock();
        let index = if help {
            0
        } else {
            chunks.iter().position(|chunk| chunk.owner == owner)?
        };
        chunks.remove(index)
    }

    /// Adds a chunk, and wakes the streams waiting for one.
    fn push(&self, chunk: PooledChunk) {
        self.chunks.lock().push_back(chunk);
        self.changed.notify_waiters();
    }

    /// Records that a root has spawned all its chunks, and wakes the streams waiting for one.
    fn root_done(&self) {
        self.roots.fetch_sub(1, Ordering::AcqRel);
        self.changed.notify_waiters();
    }

    /// Drops the chunks of the stream of `owner`, which nothing will ask for any more.
    fn purge(&self, owner: u64) {
        self.chunks.lock().retain(|chunk| chunk.owner != owner);
    }

    /// The next chunk the stream of `owner` may run, waiting while the pool has none but a root
    /// that may still add one is running. `None` once no chunk will come.
    async fn next_chunk(&self, owner: u64, help: bool) -> Option<PooledChunk> {
        loop {
            // Registered before checking, so a chunk added in between still wakes this stream.
            let changed = self.changed.notified();
            let mut changed = pin!(changed);
            changed.as_mut().enable();

            if let Some(chunk) = self.take(owner, help) {
                return Some(chunk);
            }
            // Only a stream that helps waits for others: its own root is done, so its own chunks
            // are all in the pool already. Each root it waits for is the stream of a partition
            // that polls it: DataFusion opens a partition's next file only after its current
            // stream ends, so no root waits behind a stream that waits for it.
            if !help || self.roots.load(Ordering::Acquire) == 0 {
                return None;
            }
            changed.await;
        }
    }
}

/// A partition's stream over the pool: its own file's root run, and the chunk it is running.
struct PoolStream {
    pool: Arc<ChunkPool>,
    owner: u64,
    help: bool,
    /// The root run of this stream's file, until it has spawned every chunk.
    root: Option<Run>,
    file: Arc<PooledFile>,
    /// The chunk this stream is running, and the file it belongs to.
    current: Option<(Run, Arc<PooledFile>)>,
}

impl Drop for PoolStream {
    fn drop(&mut self) {
        // A stream ends normally only once its chunks are taken; a stream dropped before that
        // leaves chunks no stream of the scan wants.
        self.pool.purge(self.owner);
        if self.root.take().is_some() {
            self.pool.root_done();
        }
    }
}

impl PoolStream {
    /// The next batch this stream yields, or `None` once the pool has nothing it may run.
    async fn next_batch(&mut self) -> DFResult<Option<RecordBatch>> {
        loop {
            if let Some((run, file)) = &mut self.current {
                match run.advance().map_err(to_df)? {
                    Progress::Batch(batch) => match (file.finish)(batch.array)? {
                        Some(batch) => return Ok(Some(batch)),
                        None => continue,
                    },
                    Progress::Spawn(scope, child) => {
                        let io = file.io.session();
                        run.admit(child, scope, io);
                    }
                    Progress::RootDone(_) if run.live_roots() > 0 => {}
                    Progress::RootDone(_) | Progress::Idle => self.current = None,
                    Progress::Waiting => {
                        future::poll_fn(|cx| run.poll_completion(cx))
                            .await
                            .map_err(to_df)?;
                    }
                }
                continue;
            }

            if let Some(root) = &mut self.root {
                match root.advance().map_err(to_df)? {
                    Progress::Batch(batch) => {
                        if let Some(batch) = (self.file.finish)(batch.array)? {
                            return Ok(Some(batch));
                        }
                    }
                    Progress::Spawn(scope, planner) => {
                        self.pool.push(PooledChunk {
                            owner: self.owner,
                            scope,
                            planner,
                            file: Arc::clone(&self.file),
                        });
                    }
                    Progress::RootDone(_) if root.live_roots() > 0 => {}
                    Progress::RootDone(_) | Progress::Idle => {
                        self.root = None;
                        self.pool.root_done();
                    }
                    Progress::Waiting => {
                        future::poll_fn(|cx| root.poll_completion(cx))
                            .await
                            .map_err(to_df)?;
                    }
                }
                continue;
            }

            let Some(chunk) = self.pool.next_chunk(self.owner, self.help).await else {
                return Ok(None);
            };
            let mut run = Run::new();
            run.admit(chunk.planner, chunk.scope, chunk.file.io.session());
            self.current = Some((run, chunk.file));
        }
    }
}

fn to_df(error: VortexError) -> DataFusionError {
    DataFusionError::External(Box::new(error))
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
