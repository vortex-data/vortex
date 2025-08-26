// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{CpuTask, Handle, IoTask, Runtime};
use async_task::Runnable;
use crossbeam_deque::{Injector, Stealer};
use futures::executor::block_on;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::Stream;
use futures::{FutureExt, StreamExt};
use smol::LocalExecutor;
use std::iter;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// A worker pool is a Vortex runtime that can be driven from multiple worker threads, typically
/// orchestrated by the system that is calling into Vortex.
///
/// Each worker makes a decision about whether to perform I/O tasks, CPU tasks, or drive the
/// underlying stream. It is therefore expected that the stream is largely a lightweight state
/// machine that alternates between spawning I/O and spawning CPU onto the runtime handle.
pub struct WorkerPool<T: Send + 'static> {
    shared: Arc<Shared<T>>,
}

impl<T: Send + 'static> WorkerPool<T> {
    pub fn drive_stream<F, S>(f: F) -> WorkerPool<T>
    where
        F: FnOnce(Handle) -> S,
        S: Stream<Item = T> + Send + 'static,
        T: Send + 'static,
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

struct Shared<T: Send + 'static> {
    scheduling: Arc<WorkStealing<Runnable>>,
    cpu: Arc<WorkStealing<CpuTask>>,
    io: Arc<WorkStealing<IoTask>>,

    /// The current count of I/O worker threads.
    active_io_workers: AtomicUsize,
    target_io_reqs_per_worker: usize, // e.g. queue len = 32, target = 8 -> 4 workers

    // The result channel.
    results: crossbeam_channel::Receiver<T>,
}

/// We implement [`Runtime`] for the worker pool's shared state, which allows us to create a handle
/// that spawns onto the shared injector queues.
///
/// Note that we _also_ implement [`Runtime`] for each individual worker, which allows us to pass
/// a handle that spawns onto a specific worker's local queues.
impl<T: Send + 'static> Runtime for Shared<T> {
    fn spawn_scheduling(&self, fut: BoxFuture<'static, ()>) {
        let injector = self.scheduling.injector.clone();
        let (runnable, task) = async_task::spawn(fut, move |r| injector.push(r));
        task.detach();
        self.scheduling.injector.push(runnable);
    }

    fn spawn_cpu(&self, task: CpuTask) {
        self.cpu.injector.push(task);
    }

    fn spawn_io(&self, mut stream: BoxStream<'static, IoTask>, _concurrency: usize) {
        // TODO(ngates): this is rather complicated for now...
        // We launch a scheduling task to push I/O requests into the work stealing queue.
        let injector = self.io.injector.clone();
        self.spawn_scheduling(
            async move {
                while let Some(task) = stream.next().await {
                    injector.push(task)
                }
            }
            .boxed(),
        )
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
        }
    }
}

pub struct Worker<T: Send + 'static> {
    shared: Arc<Shared<T>>,

    role: WorkerRole,

    scheduling: WorkStealingLocal<Runnable>,
    cpu: WorkStealingLocal<CpuTask>,
    io: WorkStealingLocal<IoTask>,
}

#[derive(Debug, Clone, Copy)]
enum WorkerRole {
    Executor, // TODO(ngates): split CPU from scheduling tasks.
    IO,
}

impl<T: Send + 'static> Worker<T> {
    fn update_role(&mut self) -> WorkerRole {
        // FIXME(ngates): this works quite well, except flume channel len requires a mutex!
        // let queue_depth = self.shared.file_io_recv.len();
        let active_workers = self.shared.active_io_workers.load(Ordering::Relaxed);
        let target_workers = 1;

        // Simple heuristic: need more I/O workers if queue is backing up
        // let target_workers = (queue_depth / self.shared.target_io_reqs_per_worker).max(1);
        // log::trace!(
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
    fn drive_io(&mut self) {
        // We should become an I/O worker until there are no in-flight requests, and no requests
        // in the queue. After that, we yield back to the worker to perform more work.
        let ex = LocalExecutor::new();

        while let Some(task) = self.io.find_task() {
            block_on(ex.run(task.run()));
        }
    }
}

/// Implementation of an iterator that actually drives the underlying runtime.
impl<T: Send + 'static> Iterator for Worker<T> {
    type Item = T;

    #[inline(never)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try to emit an item from the results channel.
            // TODO(ngates): can we essentially round-robin these? Should we even?
            match self.shared.results.try_recv() {
                Ok(item) => return Some(item),
                Err(crossbeam_channel::TryRecvError::Empty) => { /* No items, continue */ }
                Err(crossbeam_channel::TryRecvError::Disconnected) => return None,
            }

            match self.update_role() {
                WorkerRole::Executor => {
                    // We also perform scheduling if there is a task to perform.
                    // TODO(ngates): we probably shouldn't work-steal at this point?
                    while let Some(runnable) = self.scheduling.find_task() {
                        // TODO(ngates): there's no good way to tell this runnable that anything
                        //  it spawns should be sent to the current worker's local queues? I'm
                        //  sure we can figure it out.
                        runnable.run();
                        // Start the loop again to check for more work.
                        // continue;
                    }

                    if let Some(task) = self.cpu.find_task() {
                        task.run();
                        continue;
                    }
                }
                WorkerRole::IO => {
                    self.drive_io();
                    continue;
                }
            }

            // TODO(ngates): how to avoid busy looping here? Or maybe that's ok?
            std::thread::yield_now();
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

impl<T> WorkStealing<T> {
    fn new(injector: Arc<Injector<T>>) -> Self {
        Self {
            injector,
            stealers: Default::default(),
        }
    }
}

impl<T> Default for WorkStealing<T> {
    fn default() -> Self {
        Self::new(Arc::new(Injector::new()))
    }
}

impl<T> WorkStealing<T> {
    fn new_worker(self: &Arc<Self>) -> WorkStealingLocal<T> {
        let local = crossbeam_deque::Worker::new_fifo();
        let id = self.stealers.push(local.stealer()) - 1;
        WorkStealingLocal {
            id,
            global: self.clone(),
            local,
            injector: Arc::new(Injector::new()),
        }
    }
}

struct WorkStealingLocal<T> {
    id: usize,
    global: Arc<WorkStealing<T>>,
    local: crossbeam_deque::Worker<T>,
    injector: Arc<Injector<T>>, // A local injector.
}

impl<T> WorkStealingLocal<T> {
    fn find_task(&self) -> Option<T> {
        // Pop a task from the local queue, if not empty.
        self.local.pop().or_else(|| {
            // Otherwise, we need to look for a task elsewhere.
            iter::repeat_with(|| {
                // Try stealing a batch of tasks from our local injector.
                self.injector.steal_batch_and_pop(&self.local).or_else(|| {
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
            })
            // Loop while no task was stolen and any steal operation needs to be retried.
            .find(|s| !s.is_retry())
            // Extract the stolen task, if there is one.
            .and_then(|s| s.success())
        })
    }
}
