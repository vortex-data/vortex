// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::mem;
use std::sync::Arc;
use std::sync::Weak;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use async_channel::Receiver;
use async_channel::Sender;
use futures::SinkExt;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::channel::mpsc;
use futures::stream;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use vortex_error::VortexResult;
use vortex_io::request::IoService;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;
use vortex_utils::parallelism::get_available_parallelism;

use super::MapFn;
use super::PlanRoot;
use super::ScopedBatch;
use super::sort_by_scope;
use crate::planning::driver::Progress;
use crate::planning::driver::Run;

/// Roots a worker interleaves on its run by default: one to compute while another waits.
const DEFAULT_ROOTS_PER_WORKER: usize = 2;

/// Runs a plan on a pool of workers, each with a [`Run`] of its own.
///
/// Every independent child a planner emits goes to a queue the workers share. A worker takes
/// the next one whenever it runs fewer than its roots-per-worker, so a scan spreads over the
/// workers as fast as its planners declare independent work. Dependent children stay with their
/// parent on its worker.
///
/// Nothing runs until the stream is first polled. Workers are then spawned with
/// [`Handle::spawn`] as work is queued, up to the limit, so a plan with little independent work
/// spawns few tasks, and the task polling the stream does no scan work. Dropping the stream
/// cancels every worker. An error or a panic in any worker stops the others taking more work and
/// ends the stream with it.
///
/// Batches arrive as workers produce them. With [`ordered`](Self::with_ordered), the driver holds
/// them until the plan is done and yields them by file and first row.
#[derive(Clone)]
pub struct ParallelDriver {
    handle: Handle,
    workers: usize,
    roots_per_worker: usize,
    ordered: bool,
}

impl ParallelDriver {
    /// A driver spawning up to one worker per available core on `handle`, each interleaving two
    /// roots, yielding batches as they are produced.
    pub fn new(handle: Handle) -> Self {
        Self {
            handle,
            workers: get_available_parallelism().unwrap_or(1),
            roots_per_worker: DEFAULT_ROOTS_PER_WORKER,
            ordered: false,
        }
    }

    /// Runs up to `workers` workers. Values below one mean one.
    pub fn with_workers(mut self, workers: usize) -> Self {
        self.workers = workers.max(1);
        self
    }

    /// Lets each worker interleave up to `roots` roots on its run. Values below one mean one.
    pub fn with_roots_per_worker(mut self, roots: usize) -> Self {
        self.roots_per_worker = roots.max(1);
        self
    }

    /// Whether to yield batches by file and first row, holding them until the plan is done.
    pub fn with_ordered(mut self, ordered: bool) -> Self {
        self.ordered = ordered;
        self
    }

    /// Runs `root` and everything it emits, streaming the batches of its morsels.
    pub fn execute(&self, root: PlanRoot) -> BoxStream<'static, VortexResult<ScopedBatch>> {
        self.execute_with(root, Arc::new(Ok))
    }

    /// Runs `root`, applying `map_fn` to each batch on the worker that produced it.
    pub fn execute_with<A: Send + 'static>(
        &self,
        root: PlanRoot,
        map_fn: MapFn<A>,
    ) -> BoxStream<'static, VortexResult<ScopedBatch<A>>> {
        let driver = self.clone();
        let batches = stream::once(async move { driver.start(root, map_fn) }).flatten();
        if !self.ordered {
            return batches.boxed();
        }
        stream::once(async move {
            let mut held: Vec<ScopedBatch<A>> = batches.try_collect().await?;
            sort_by_scope(&mut held);
            VortexResult::Ok(stream::iter(held.into_iter().map(Ok)))
        })
        .try_flatten()
        .boxed()
    }

    /// Queues `root`, which spawns the first worker, and returns the workers' batches, in the
    /// order they are produced.
    fn start<A: Send + 'static>(
        self,
        root: PlanRoot,
        map_fn: MapFn<A>,
    ) -> BoxStream<'static, VortexResult<ScopedBatch<A>>> {
        // Unbounded: a planner that spawns, such as a scan's chunk planner, runs ahead and queues
        // every independent child at once. Each child is a small planner that has done no IO yet,
        // and bounding the queue would park the spawning worker, which deadlocks once every
        // worker holds a spawning root.
        let (queue, take) = async_channel::unbounded();
        let (out, batches) = mpsc::channel(self.workers);
        // Owned by the stream alone: dropping the stream drops the tasks, which cancels them.
        let tasks = Arc::new(Mutex::new(Vec::new()));
        let shared = Arc::new(Shared {
            handle: self.handle.clone(),
            io: Arc::clone(&root.io),
            queue,
            take,
            out,
            map_fn,
            tasks: Arc::downgrade(&tasks),
            stopped: AtomicBool::new(false),
            pending: AtomicUsize::new(0),
            spawned: AtomicUsize::new(0),
            workers: self.workers,
            roots_per_worker: self.roots_per_worker,
        });
        shared.push(root);
        // The workers hold the only senders from here, so the batches end once every worker has.
        drop(shared);
        // Awaiting the workers once the batches end re-raises a worker's panic. Every worker is
        // spawned by the root or by another worker before that one ends, so all are listed.
        let joined = stream::once(async move {
            let tasks = mem::take(&mut *tasks.lock());
            futures::future::join_all(tasks).await;
        })
        .filter_map(|()| async { None });
        batches.chain(joined).boxed()
    }
}

/// What the workers of one plan share.
struct Shared<A> {
    handle: Handle,
    /// The plan's byte source, which every independent child reads.
    io: Arc<dyn IoService>,
    /// Roots no worker has taken yet. Closed once nothing is pending, which stops the workers.
    queue: Sender<PlanRoot>,
    take: Receiver<PlanRoot>,
    out: mpsc::Sender<VortexResult<ScopedBatch<A>>>,
    map_fn: MapFn<A>,
    /// The stream's list of worker tasks; gone once the stream is dropped.
    tasks: Weak<Mutex<Vec<Task<()>>>>,
    /// Set when a worker fails, panics or finds the stream dropped: no worker takes another root.
    /// Closing the queue alone would still hand out the roots already in it.
    stopped: AtomicBool,
    /// Roots queued or running anywhere. A root is counted before its parent can finish, so this
    /// reaches zero only once the plan is done.
    pending: AtomicUsize,
    /// Workers spawned so far.
    spawned: AtomicUsize,
    workers: usize,
    roots_per_worker: usize,
}

impl<A: Send + 'static> Shared<A> {
    /// Queues `root`, spawning a worker for it while fewer than the limit are running.
    fn push(self: &Arc<Self>, root: PlanRoot) {
        self.pending.fetch_add(1, Ordering::AcqRel);
        // The queue only closes once nothing is pending, and this root's parent still is.
        drop(self.queue.try_send(root));
        let below_limit = self
            .spawned
            .try_update(Ordering::AcqRel, Ordering::Acquire, |spawned| {
                (spawned < self.workers).then_some(spawned + 1)
            })
            .is_ok();
        if !below_limit {
            return;
        }
        let Some(tasks) = self.tasks.upgrade() else {
            // The stream was dropped: nothing will read what a new worker produces.
            return;
        };
        let worker = Worker {
            shared: Arc::clone(self),
            out: self.out.clone(),
        };
        tasks.lock().push(self.handle.spawn(worker.run()));
    }

    fn finish(&self) {
        if self.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.queue.close();
        }
    }

    /// Stops every worker taking more roots. Each finishes only what it already runs.
    fn stop(&self) {
        self.stopped.store(true, Ordering::Release);
        self.queue.close();
    }

    /// The next queued root, if any and the plan has not stopped.
    fn try_take(&self) -> Option<PlanRoot> {
        if self.stopped.load(Ordering::Acquire) {
            return None;
        }
        self.take.try_recv().ok()
    }

    /// Waits for the next root, or `None` once the plan is done or has stopped.
    async fn take(&self) -> Option<PlanRoot> {
        let root = self.take.recv().await.ok()?;
        (!self.stopped.load(Ordering::Acquire)).then_some(root)
    }
}

/// Stops the plan unless disarmed: dropped without disarming, the worker panicked or was
/// cancelled, and the other workers must not wait for roots that will never finish.
struct StopOnDrop<A: Send + 'static> {
    shared: Arc<Shared<A>>,
    armed: bool,
}

impl<A: Send + 'static> Drop for StopOnDrop<A> {
    fn drop(&mut self) {
        if self.armed {
            self.shared.stop();
        }
    }
}

struct Worker<A> {
    shared: Arc<Shared<A>>,
    out: mpsc::Sender<VortexResult<ScopedBatch<A>>>,
}

impl<A: Send + 'static> Worker<A> {
    async fn run(mut self) {
        let mut guard = StopOnDrop {
            shared: Arc::clone(&self.shared),
            armed: true,
        };
        if let Err(error) = self.drive().await {
            // Stop every worker, then report the error unless the stream was dropped.
            self.shared.stop();
            drop(self.out.send(Err(error)).await);
        }
        guard.armed = false;
    }

    async fn drive(&mut self) -> VortexResult<()> {
        let shared = Arc::clone(&self.shared);
        let mut run = Run::new();
        loop {
            while run.live_roots() < shared.roots_per_worker {
                match shared.try_take() {
                    Some(root) => root.admit_to(&mut run),
                    None => break,
                }
            }
            if run.live_roots() == 0 {
                match shared.take().await {
                    Some(root) => {
                        root.admit_to(&mut run);
                        continue;
                    }
                    // The plan is done, or another worker stopped it.
                    None => return Ok(()),
                }
            }
            match run.advance()? {
                Progress::Batch(batch) => {
                    let batch = ScopedBatch {
                        scope: batch.scope,
                        batch: (shared.map_fn)(batch.array)?,
                    };
                    if self.out.send(Ok(batch)).await.is_err() {
                        // The stream was dropped.
                        shared.stop();
                        return Ok(());
                    }
                }
                Progress::Spawn(scope, planner) => {
                    shared.push(PlanRoot::new(scope, planner, Arc::clone(&shared.io)));
                }
                Progress::RootDone(_) => shared.finish(),
                Progress::Idle => {}
                Progress::Waiting => {
                    if run.live_roots() < shared.roots_per_worker
                        && let Some(root) = shared.try_take()
                    {
                        root.admit_to(&mut run);
                        continue;
                    }
                    future::poll_fn(|cx| run.poll_completion(cx)).await?;
                }
            }
        }
    }
}
