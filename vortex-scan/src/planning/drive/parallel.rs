// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future;
use std::sync::Arc;
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
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;
use vortex_io::runtime::Task;
use vortex_utils::parallelism::get_available_parallelism;

use super::MapFn;
use super::PlanDriver;
use super::PlanRoot;
use super::Roots;
use super::ScopedBatch;
use super::sort_by_scope;
use crate::planning::driver::Progress;

/// Roots a worker interleaves on its run by default: one to compute while another waits.
const DEFAULT_ROOTS_PER_WORKER: usize = 2;

/// Runs a plan on a pool of workers, each with a [`Run`](crate::planning::driver::Run) of its
/// own.
///
/// Every independent child a planner emits goes to a queue the workers share. A worker takes
/// the next one whenever it runs fewer than its roots-per-worker, so a scan spreads over the
/// workers as fast as its planners declare independent work, whether that is the chunks of one
/// file or the files of a table. Dependent children stay with their parent on its worker.
///
/// The first worker runs in the returned stream, and the rest are spawned only once a planner
/// emits an independent child, so a plan with nothing to spread runs as on a sequential driver.
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
    /// A driver spawning its workers on `handle`: one per available core, each interleaving two
    /// roots, yielding batches as they are produced.
    pub fn new(handle: Handle) -> Self {
        Self {
            handle,
            workers: get_available_parallelism().unwrap_or(1),
            roots_per_worker: DEFAULT_ROOTS_PER_WORKER,
            ordered: false,
        }
    }

    /// Runs `workers` workers. Values below one mean one.
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

    /// Runs `root`, applying `map_fn` to each batch on the worker that produced it.
    pub fn execute_with<A: Send + 'static>(
        &self,
        root: PlanRoot,
        map_fn: MapFn<A>,
    ) -> BoxStream<'static, VortexResult<ScopedBatch<A>>> {
        let (queue, take) = async_channel::unbounded();
        let shared = Arc::new(Shared {
            queue,
            pending: AtomicUsize::new(1),
        });
        // An unbounded queue whose receiver is alive accepts every send.
        drop(shared.queue.try_send(root));
        let (out, batches) = mpsc::channel(self.workers);
        // The first worker runs in the stream itself. It spawns the others only once the plan
        // emits independent work, so a plan that is one chain of dependent steps, such as an
        // index search, costs no more than on a sequential driver.
        let first = Worker {
            shared,
            take,
            out,
            map_fn,
            roots_per_worker: self.roots_per_worker,
            helpers: Some(Helpers {
                handle: self.handle.clone(),
                count: self.workers - 1,
            }),
            spawned: Vec::new(),
        };
        // Dropping the stream early drops the first worker and the helpers it spawned, which
        // cancels them.
        let joined = stream::once(first.run()).filter_map(|()| async { None });
        let batches = stream::select(batches, joined);
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
}

impl PlanDriver for ParallelDriver {
    fn execute(&self, root: PlanRoot) -> BoxStream<'static, VortexResult<ScopedBatch>> {
        self.execute_with(root, Arc::new(Ok))
    }
}

/// What the workers of one plan share.
struct Shared {
    /// Roots no worker has taken yet. Closed once nothing is pending, which stops the workers.
    queue: Sender<PlanRoot>,
    /// Roots queued or running anywhere. A root is counted before its parent can finish, so this
    /// reaches zero only once the plan is done.
    pending: AtomicUsize,
}

impl Shared {
    fn push(&self, root: PlanRoot) {
        self.pending.fetch_add(1, Ordering::AcqRel);
        // The queue only closes once nothing is pending, and this root's parent still is.
        drop(self.queue.try_send(root));
    }

    fn finish(&self) {
        if self.pending.fetch_sub(1, Ordering::AcqRel) == 1 {
            self.queue.close();
        }
    }
}

/// The workers the first worker spawns once the plan emits independent work.
struct Helpers {
    handle: Handle,
    count: usize,
}

struct Worker<A> {
    shared: Arc<Shared>,
    take: Receiver<PlanRoot>,
    out: mpsc::Sender<VortexResult<ScopedBatch<A>>>,
    map_fn: MapFn<A>,
    roots_per_worker: usize,
    /// Only on the first worker, until it spawns them.
    helpers: Option<Helpers>,
    spawned: Vec<Task<()>>,
}

impl<A: Send + 'static> Worker<A> {
    async fn run(mut self) {
        if let Err(error) = self.drive().await {
            // Stop every worker, then report the error unless the stream was dropped.
            self.shared.queue.close();
            drop(self.out.send(Err(error)).await);
            return;
        }
        // Let the helpers' batches through before awaiting them, which re-raises a panic.
        drop(self.out);
        futures::future::join_all(self.spawned).await;
    }

    fn spawn_helpers(&mut self) {
        let Some(helpers) = self.helpers.take() else {
            return;
        };
        self.spawned = (0..helpers.count)
            .map(|_| {
                helpers.handle.spawn(
                    Worker {
                        shared: Arc::clone(&self.shared),
                        take: self.take.clone(),
                        out: self.out.clone(),
                        map_fn: Arc::clone(&self.map_fn),
                        roots_per_worker: self.roots_per_worker,
                        helpers: None,
                        spawned: Vec::new(),
                    }
                    .run(),
                )
            })
            .collect();
    }

    async fn drive(&mut self) -> VortexResult<()> {
        let mut roots = Roots::default();
        loop {
            while roots.run.live_roots() < self.roots_per_worker {
                match self.take.try_recv() {
                    Ok(root) => roots.admit(root),
                    Err(_) => break,
                }
            }
            if roots.run.live_roots() == 0 {
                match self.take.recv().await {
                    Ok(root) => {
                        roots.admit(root);
                        continue;
                    }
                    // Closed: the plan is done, or another worker failed.
                    Err(_) => return Ok(()),
                }
            }
            match roots.run.advance()? {
                Progress::Batch(batch) => {
                    let batch = ScopedBatch {
                        scope: batch.scope,
                        batch: (self.map_fn)(batch.array)?,
                    };
                    if self.out.send(Ok(batch)).await.is_err() {
                        // The stream was dropped.
                        self.shared.queue.close();
                        return Ok(());
                    }
                }
                Progress::Spawn(spawned) => {
                    self.shared.push(roots.root_of(spawned)?);
                    self.spawn_helpers();
                }
                Progress::RootDone(root) => {
                    roots.finish(root);
                    self.shared.finish();
                }
                Progress::Idle => {}
                Progress::Waiting => {
                    if roots.run.live_roots() < self.roots_per_worker
                        && let Ok(root) = self.take.try_recv()
                    {
                        roots.admit(root);
                        continue;
                    }
                    future::poll_fn(|cx| roots.run.poll_completion(cx)).await?;
                }
            }
        }
    }
}
