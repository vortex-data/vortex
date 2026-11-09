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
use vortex_io::request::IoService;
use vortex_io::runtime::Handle;
use vortex_utils::parallelism::get_available_parallelism;

use super::MapFn;
use super::PlanDriver;
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
/// Batches arrive as workers produce them. With [`ordered`](Self::with_ordered), the driver holds
/// them until the plan is done and yields them by file and first row.
///
/// The workers are spawned with [`Handle::spawn`], so the task polling the stream does no scan
/// work.
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
        let io = Arc::clone(&root.io);
        let (queue, take) = async_channel::unbounded();
        let shared = Arc::new(Shared {
            queue,
            pending: AtomicUsize::new(1),
        });
        // An unbounded queue whose receiver is alive accepts every send.
        drop(shared.queue.try_send(root));
        let (out, batches) = mpsc::channel(self.workers);
        let tasks: Vec<_> = (0..self.workers)
            .map(|_| {
                let worker = Worker {
                    shared: Arc::clone(&shared),
                    take: take.clone(),
                    out: out.clone(),
                    map_fn: Arc::clone(&map_fn),
                    io: Arc::clone(&io),
                    roots_per_worker: self.roots_per_worker,
                };
                self.handle.spawn(worker.run())
            })
            .collect();
        drop(out);
        // Awaiting the workers once the stream ends re-raises a worker's panic, and dropping the
        // stream early drops them, which cancels them.
        let joined = stream::once(futures::future::join_all(tasks)).filter_map(|_| async { None });
        let batches = batches.chain(joined);
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

struct Worker<A> {
    shared: Arc<Shared>,
    take: Receiver<PlanRoot>,
    out: mpsc::Sender<VortexResult<ScopedBatch<A>>>,
    map_fn: MapFn<A>,
    /// The plan's byte source, which every independent child reads.
    io: Arc<dyn IoService>,
    roots_per_worker: usize,
}

impl<A: Send + 'static> Worker<A> {
    async fn run(mut self) {
        if let Err(error) = self.drive().await {
            // Stop every worker, then report the error unless the stream was dropped.
            self.shared.queue.close();
            drop(self.out.send(Err(error)).await);
        }
    }

    async fn drive(&mut self) -> VortexResult<()> {
        let mut run = Run::new();
        loop {
            while run.live_roots() < self.roots_per_worker {
                match self.take.try_recv() {
                    Ok(root) => root.admit_to(&mut run),
                    Err(_) => break,
                }
            }
            if run.live_roots() == 0 {
                match self.take.recv().await {
                    Ok(root) => {
                        root.admit_to(&mut run);
                        continue;
                    }
                    // Closed: the plan is done, or another worker failed.
                    Err(_) => return Ok(()),
                }
            }
            match run.advance()? {
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
                Progress::Spawn(scope, planner) => {
                    self.shared.push(PlanRoot::new(scope, planner, Arc::clone(&self.io)));
                }
                Progress::RootDone(_) => self.shared.finish(),
                Progress::Idle => {}
                Progress::Waiting => {
                    if run.live_roots() < self.roots_per_worker
                        && let Ok(root) = self.take.try_recv()
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
