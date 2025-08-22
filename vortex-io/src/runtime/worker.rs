// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{CpuTask, FileIoRequest, Handle, Runtime};
use async_task::Runnable;
use crossbeam_deque::{Injector, Stealer};
use flume::Receiver;
use futures::{pin_mut, Stream};
use futures_util::future::FutureObj;
use futures_util::stream::FuturesUnordered;
use futures_util::task::noop_waker_ref;
use futures_util::StreamExt;
use parking_lot::RwLock;
use smol::unblock;
use std::iter;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::task::LocalSet;
use vortex_buffer::ByteBufferMut;
use vortex_error::{VortexError, VortexExpect};

impl Runtime {
    /// Returns a worker pool that can be used to drive the Runtime and in the process emit
    /// items from the stream.
    pub fn drive_stream_on_pool<F, S, R>(self, f: F) -> WorkerPool<R>
    where
        F: FnOnce(Handle) -> S,
        S: Stream<Item = R> + Unpin + Send + 'static,
        R: Send + 'static,
    {
        let handle = self.handle.clone();
        let stream = f(handle.clone());

        let scheduling_injector = Arc::new(Injector::<Runnable>::new());

        // We create a channel for the output results and spawn a detached task to populate it.
        // This will be driven by the scheduler.
        let (send, recv) = crossbeam_channel::unbounded::<R>();
        let results_fut = async move {
            pin_mut!(stream);
            while let Some(item) = stream.next().await {
                if let Err(e) = send.send(item) {
                    log::trace!("All workers disconnected: {}", e);
                    break;
                }
            }
        };
        let scheduling_injector2 = scheduling_injector.clone();
        let (results_runnable, results_task) =
            async_task::spawn(results_fut, move |r| scheduling_injector2.push(r));
        results_task.detach();
        scheduling_injector.push(results_runnable);

        WorkerPool {
            shared: Arc::new(Shared {
                recv_sched: self.sched_recv,
                recv_cpu: self.cpu_recv,
                recv_io: self.io_recv,
                scheduling: Arc::new(WorkStealing::new(scheduling_injector)),
                cpu: Default::default(),
                active_io_workers: Default::default(),
                target_io_reqs_per_worker: 32,
                results: recv,
            }),
        }
    }
}

/// A worker pool is way of driving a Vortex runtime from multiple worker threads, typically
/// orchestrated by the system that is calling into Vortex.
///
/// Each worker makes a decision about whether to perform I/O tasks, CPU tasks, or drive the
/// underlying stream. It is therefore expected that the stream is largely a lightweight state
/// machine that alternates between spawning I/O and spawning CPU onto the runtime handle.
pub struct WorkerPool<T: Send + 'static> {
    shared: Arc<Shared<T>>,
}

struct Shared<T: Send + 'static> {
    // The recver side of the runtime.
    recv_sched: Receiver<FutureObj<'static, ()>>,
    recv_cpu: Receiver<CpuTask>,
    recv_io: Receiver<FileIoRequest>,

    // The global injector for scheduling tasks.
    scheduling: Arc<WorkStealing<Runnable>>,
    cpu: Arc<WorkStealing<CpuTask>>,

    /// The current count of I/O worker threads.
    active_io_workers: AtomicUsize,
    target_io_reqs_per_worker: usize, // e.g. queue len = 32, target = 8 -> 4 workers

    // The result channel.
    results: crossbeam_channel::Receiver<T>,
}

impl<T: Send + 'static> WorkerPool<T> {
    pub fn new_worker(&self) -> Worker<T> {
        let scheduling = self.shared.scheduling.new_worker();

        let io_runtime = tokio::runtime::Builder::new_current_thread()
            .thread_name(format!("vortex-worker-{}", scheduling.id))
            .enable_all()
            .build()
            .vortex_expect("Failed to create worker I/O runtime");

        Worker {
            shared: self.shared.clone(),
            role: WorkerRole::Executor,
            scheduling,
            cpu: self.shared.cpu.new_worker(),
            io_runtime,
        }
    }
}

pub struct Worker<T: Send + 'static> {
    shared: Arc<Shared<T>>,

    role: WorkerRole,

    scheduling: WorkStealingLocal<Runnable>,
    cpu: WorkStealingLocal<CpuTask>,

    // FIXME(ngates): we need to share a pool of workers that perform blocking reads...
    io_runtime: tokio::runtime::Runtime,
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
        let file_io_recv = self.shared.recv_io.clone();
        self.io_runtime
            .block_on(LocalSet::new().run_until(async move {
                // A no-op context.
                let mut cx = Context::from_waker(noop_waker_ref());

                // Convert receiver to stream for easier polling
                let mut file_io_stream = file_io_recv.into_stream();

                // Create a FuturesUnordered to manage concurrent blocking operations
                let mut inflight = FuturesUnordered::new();

                loop {
                    // Try to fill up to our concurrency limit with new requests
                    let mut got_new_request = false;

                    while inflight.len() < 16 {
                        match file_io_stream.poll_next_unpin(&mut cx) {
                            Poll::Ready(Some(req)) => {
                                got_new_request = true;

                                // Spawn a new blocking operation
                                let fut = unblock(move || {
                                    let mut buffer = ByteBufferMut::with_capacity_aligned(
                                        req.length,
                                        req.alignment,
                                    );
                                    unsafe { buffer.set_len(req.length) };
                                    match req.file.read_exact_at(&mut buffer, req.offset) {
                                        Ok(()) => req.resolve(Ok(buffer.freeze())),
                                        Err(e) => req.resolve(Err(VortexError::from(e))),
                                    }
                                });
                                inflight.push(fut);
                            }
                            Poll::Ready(None) => {
                                // Channel closed
                                break;
                            }
                            Poll::Pending => {
                                // No more requests available right now
                                break;
                            }
                        }
                    }

                    // If we have pending operations, wait for at least one to complete
                    if !inflight.is_empty() {
                        while let Some(()) = inflight.next().await {}
                    } else if !got_new_request {
                        // No pending operations and no new requests available - terminate
                        return;
                    }
                }
            }));
    }
}

/// Implementation of an iterator that actually drives the underlying runtime.
impl<T: Send + 'static> Iterator for Worker<T> {
    type Item = T;

    #[inline(never)]
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try to emit an item from the results channel.
            match self.shared.results.try_recv() {
                Ok(item) => return Some(item),
                Err(crossbeam_channel::TryRecvError::Empty) => { /* No items, continue */ }
                Err(crossbeam_channel::TryRecvError::Disconnected) => return None,
            }

            // Move globally pending tasks into our work-stealing queues.
            while let Ok(fut) = self.shared.recv_sched.try_recv() {
                let injector = self.shared.scheduling.injector.clone();
                let (runnable, task) = async_task::spawn(fut, move |r| injector.push(r));
                // TODO(ngates): I'm not sure about detaching here...
                task.detach();
                self.shared.scheduling.injector.push(runnable);
            }
            while let Ok(task) = self.shared.recv_cpu.try_recv() {
                self.shared.cpu.injector.push(task);
            }

            // Try to drive the scheduler if there is a task to perform.
            // TODO(ngates): we probably shouldn't work-steal at this point?
            if let Some(runnable) = self.scheduling.find_task() {
                runnable.run();
                // Start the loop again to check for more work.
                continue;
            }

            match self.update_role() {
                WorkerRole::Executor => {
                    if let Some(task) = self.cpu.find_task() {
                        task.run();
                        continue;
                    }
                }
                WorkerRole::IO => {
                    self.drive_io();
                }
            }

            // TODO(ngates): how to avoid busy looping here? Or maybe that's ok?
            // std::thread::yield_now();
        }
    }
}

struct WorkStealing<T> {
    injector: Arc<Injector<T>>,
    stealers: RwLock<Vec<Stealer<T>>>,
}

impl<T> WorkStealing<T> {
    fn new(injector: Arc<Injector<T>>) -> Self {
        Self {
            injector,
            stealers: RwLock::new(Vec::new()),
        }
    }
}

impl<T> Default for WorkStealing<T> {
    fn default() -> Self {
        Self {
            injector: Arc::new(Injector::new()),
            stealers: RwLock::new(Vec::new()),
        }
    }
}

impl<T> WorkStealing<T> {
    fn new_worker(self: &Arc<Self>) -> WorkStealingLocal<T> {
        let local = crossbeam_deque::Worker::new_fifo();
        let mut stealers = self.stealers.write();
        let id = stealers.len(); // Grab our ID while we hold the write lock.
        stealers.push(local.stealer());
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
                            .read()
                            .iter()
                            .map(|s| s.steal())
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
