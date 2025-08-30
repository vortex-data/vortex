// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{CpuTask, Handle, IoTask, Runtime};
use async_compat::Compat;
use async_task::Runnable;
use crossbeam_deque::{Injector, Stealer};
use futures::executor::block_on;
use futures::future::BoxFuture;
use futures::stream::{BoxStream, FuturesUnordered};
use futures::Stream;
use futures::{FutureExt, StreamExt};
use itertools::Itertools;
use smol::lock::{Semaphore, SemaphoreGuardArc};
use smol::LocalExecutor;
use std::any::type_name;
use std::fmt::{Debug, Formatter};
use std::iter;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A worker pool is a Vortex runtime that can be driven from multiple worker threads, typically
/// orchestrated by the system that is calling into Vortex.
///
/// Each worker makes a decision about whether to perform I/O tasks, CPU tasks, or drive the
/// underlying stream. It is therefore expected that the stream is largely a lightweight state
/// machine that alternates between spawning I/O and spawning CPU onto the runtime handle.
pub struct WorkerPool<T: Send> {
    shared: Arc<Shared<T>>,
}

impl<T: Send> WorkerPool<T> {
    pub fn drive_stream<'rt, F, S>(f: F) -> WorkerPool<T>
    where
        F: FnOnce(Handle<'rt>) -> S,
        S: Stream<Item = T> + Send + 'rt,
        T: 'rt,
    {
        let (result_tx, result_rx) = crossbeam_channel::unbounded();

        let shared = Arc::new(Shared {
            scheduling: Arc::new(WorkStealing::default()),
            cpu: Arc::new(WorkStealing::default()),
            io: Arc::new(WorkStealing::default()),
            active_io_workers: AtomicUsize::new(0),
            target_io_reqs_per_worker: 8,
            results: result_rx,
        });

        let handle = Handle(shared.clone());
        let stream = f(handle.clone());

        // Spawn a task to drive the stream and send results to the result channel.
        shared.spawn_scheduling(
            async move {
                futures::pin_mut!(stream);
                while let Some(item) = stream.next().await {
                    // Ignore send errors, which happen if the receiver is dropped.
                    let _ = result_tx.send(item);
                }
            }
            .boxed(),
        );

        WorkerPool { shared }
    }
}

struct Shared<T: Send> {
    scheduling: Arc<WorkStealing<Runnable>>,
    cpu: Arc<WorkStealing<CpuTask>>,
    io: Arc<WorkStealing<Guarded<IoTask>>>,

    /// The current count of I/O worker threads.
    active_io_workers: AtomicUsize,
    target_io_reqs_per_worker: usize, // e.g. queue len = 32, target = 8 -> 4 workers

    // The result channel.
    results: crossbeam_channel::Receiver<T>,
}

/// A wrapper around T that holds a semaphore permit.
struct Guarded<T> {
    inner: T,
    guard: SemaphoreGuardArc,
}

/// We implement [`Runtime`] for the worker pool's shared state, which allows us to create a handle
/// that spawns onto the shared injector queues.
///
/// Note that we _also_ implement [`Runtime`] for each individual worker, which allows us to pass
/// a handle that spawns onto a specific worker's local queues.
impl<'rt, T: Send> Runtime<'rt> for Shared<T> {
    fn spawn_scheduling(&self, fut: BoxFuture<'rt, ()>) {
        let injector = self.scheduling.injector.clone();

        // SAFETY: We know that the future is `Send`, and we know that schedule is `Send` + `Sync`.
        //  Really, we are just avoiding the 'static bound on the future because we know the handle
        //  lifetime will out-live the future.
        // TOOD(ngates): we should create custom function with the Send + Sync bounds,
        let (runnable, task) =
            unsafe { async_task::spawn_unchecked(fut, move |r| injector.push(r)) };

        task.detach();
        runnable.schedule();
    }

    fn spawn_cpu(&self, task: CpuTask) {
        self.cpu.injector.push(task);
    }

    fn spawn_io(&self, mut stream: BoxStream<'rt, IoTask>, concurrency: usize) {
        // This is quite tricky to get right. Essentially, we want to defer pulling from the
        // stream for as long as possible because that allows us to perform better coalescing of
        // I/O requests. We use the given concurrency parameter to achieve this.
        let semaphore = Arc::new(Semaphore::new(concurrency));
        let injector = self.io.injector.clone();
        self.spawn_scheduling(
            async move {
                while let Some(task) = stream.next().await {
                    let guard = semaphore.acquire_arc().await;
                    injector.push(Guarded { inner: task, guard })
                }
            }
            .boxed(),
        );
    }
}

impl<T: Send + 'static> WorkerPool<T> {
    pub fn new_worker(&self) -> Worker<T> {
        let scheduling = self.shared.scheduling.new_worker();

        Worker {
            shared: self.shared.clone(),
            role: WorkerRole::Executor,
            scheduling,
            cpu: self.shared.cpu.new_worker(),
            io: self.shared.io.new_worker(),
            io_executor: LocalExecutor::new(),
        }
    }
}

pub struct Worker<T: Send + 'static> {
    shared: Arc<Shared<T>>,

    role: WorkerRole,

    scheduling: WorkStealingLocal<Runnable>,
    cpu: WorkStealingLocal<CpuTask>,
    io: WorkStealingLocal<Guarded<IoTask>>,

    io_executor: LocalExecutor<'static>,
}

#[derive(Debug, Clone, Copy)]
enum WorkerRole {
    Executor, // TODO(ngates): split CPU from scheduling tasks.
    IO,
}

impl<T: Send + 'static> Worker<T> {
    fn update_role(&mut self) -> WorkerRole {
        // FIXME(ngates): this works quite well, except kanal channel len requires a mutex!
        // We should maintain our own atomic counter of queue depth.
        // let queue_depth = self.shared.io.injector.len();
        let active_workers = self.shared.active_io_workers.load(Ordering::Relaxed);

        // Simple heuristic: need more I/O workers if queue is backing up
        let target_workers = 1;
        // let target_workers = (queue_depth / self.shared.target_io_reqs_per_worker).max(1);
        // log::info!(
        //     "Queue depth: {}, Active I/O workers: {}, Target I/O workers: {}",
        //     queue_depth,
        //     active_workers,
        //     target_workers
        // );

        match self.role {
            WorkerRole::Executor => {
                if active_workers < target_workers {
                    // Try to increment atomically
                    self.shared.active_io_workers.fetch_add(1, Ordering::AcqRel);
                    // Upgrade to I/O role
                    self.role = WorkerRole::IO;
                }
            }
            WorkerRole::IO => {
                if active_workers > target_workers {
                    // Try to decrement atomically
                    self.shared.active_io_workers.fetch_sub(1, Ordering::AcqRel);
                    // Downgrade to executor role
                    self.role = WorkerRole::Executor;
                }
            }
        }

        self.role
    }

    /// Perform the role of the I/O driver.
    fn drive_io(&mut self, tasks: Vec<Guarded<IoTask>>) {
        // We should become an I/O worker until there are no in-flight requests, and no requests
        // in the queue. After that, we yield back to the worker to perform more work.

        // log::trace!("Driving {} I/O tasks", tasks.len());
        let mut handles = Vec::with_capacity(tasks.len());
        self.io_executor.spawn_many(
            tasks.into_iter().map(|task| async move {
                Compat::new(task.inner.run_send()).await;
                drop(task.guard)
            }),
            &mut handles,
        );

        block_on(
            self.io_executor
                .run(FuturesUnordered::from_iter(handles).collect::<()>()),
        );
    }
}

/// Implementation of an iterator that actually drives the underlying runtime.
impl<T: Send + 'static> Iterator for Worker<T> {
    type Item = T;

    #[inline(never)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // 1. FIRST: Try to execute scheduling tasks
            //    These spawn new work and keep pipelines full
            if let Some(runnable) = self.scheduling.find_task() {
                runnable.run();
                continue; // Keep pipeline primed before emitting
            }

            let role = self.update_role();

            if matches!(role, WorkerRole::IO) {
                // 2. SECOND: Check for I/O work (if role appropriate)
                //    Keep I/O pipeline moving
                let tasks = iter::from_fn(|| self.io.find_task()).collect_vec();
                if !tasks.is_empty() {
                    self.drive_io(tasks);
                    continue;
                }
            }

            // 3. THIRD: Emit results prior to performing our own CPU work.
            //    This hands control back to the caller
            match self.shared.results.try_recv() {
                Ok(item) => return Some(item),
                Err(crossbeam_channel::TryRecvError::Empty) => {}
                Err(crossbeam_channel::TryRecvError::Disconnected) => return None,
            }

            if matches!(role, WorkerRole::Executor) {}
            // 4. FOURTH: Execute CPU tasks
            //    Complete in-flight computations
            log::trace!("Driving CPU: {:?}", self.scheduling);
            if let Some(task) = self.cpu.find_task() {
                task.run();
                continue;
            }

            std::thread::yield_now()
        }
    }
}

struct WorkStealing<T> {
    injector: Arc<Injector<T>>,
    // We use BoxCar as a lock-free append-only vec for stealers. This allows us to defer
    // creating new workers, without paying the cost of a read-lock every time we try to steal.
    // TODO(ngates): look into BoxCar for lock-free append-only vec.
    stealers: boxcar::Vec<Stealer<T>>,
}

impl<T> Default for WorkStealing<T> {
    fn default() -> Self {
        Self {
            injector: Arc::new(Injector::new()),
            stealers: Default::default(),
        }
    }
}

impl<T> WorkStealing<T> {
    fn new_worker(self: &Arc<Self>) -> WorkStealingLocal<T> {
        let local = crossbeam_deque::Worker::new_fifo();
        let id = self.stealers.push(local.stealer());
        WorkStealingLocal {
            id,
            global: self.clone(),
            local,
        }
    }
}

struct WorkStealingLocal<T> {
    id: usize,
    global: Arc<WorkStealing<T>>,
    local: crossbeam_deque::Worker<T>,
}

impl<T> Debug for WorkStealingLocal<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct(&format!("WorkStealingLocal<{}>", type_name::<T>()))
            .field("id", &self.id)
            .field("local_len", &self.local.len())
            .field("injector_len", &self.global.injector.len())
            .field(
                "stealers_lens",
                &self
                    .global
                    .stealers
                    .iter()
                    .map(|(_idx, s)| s.len())
                    .collect_vec(),
            )
            .finish()
    }
}

impl<T> WorkStealingLocal<T> {
    fn find_task(&self) -> Option<T> {
        // Pop a task from the local queue, if not empty.
        self.local.pop().or_else(|| {
            // Otherwise, we need to look for a task elsewhere.
            iter::repeat_with(|| {
                // Try stealing a batch of tasks from the global queue.
                self.global
                    .injector
                    .steal_batch_and_pop(&self.local)
                    // Or try stealing a task from one of the other threads.
                    .or_else(|| {
                        self.global
                            .stealers
                            .iter()
                            .map(|(_idx, s)| s.steal())
                            .collect()
                    })
            })
            // Loop while no task was stolen and any steal operation needs to be retried.
            .find(|s| !s.is_retry())
            // Extract the stolen task, if there is one.
            .and_then(|s| s.success())
        })
    }
}
