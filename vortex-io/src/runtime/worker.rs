// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{FileIoRequest, Runtime};
use async_task::Runnable;
use crossbeam_deque::{Injector, Stealer};
use flume::Receiver;
use futures::executor::block_on;
use futures::{pin_mut, Stream};
use futures_util::stream::FuturesUnordered;
use futures_util::task::noop_waker_ref;
use futures_util::StreamExt;
use parking_lot::RwLock;
use smol::{unblock, Executor};
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
    pub fn drive_stream_on_pool<T: Send + 'static>(
        self,
        stream: impl Stream<Item = T> + Send + 'static,
    ) -> WorkerPool<T> {
        // We create a channel for the output results and spawn a detached task to populate it.
        let (send, recv) = crossbeam_channel::unbounded::<T>();
        self.executor
            .spawn(async move {
                pin_mut!(stream);
                while let Some(item) = stream.next().await {
                    if let Err(e) = send.send(item) {
                        log::trace!("All workers disconnected: {}", e);
                        break;
                    }
                }
            })
            .detach();

        WorkerPool {
            shared: Arc::new(Shared {
                next_worker_id: Default::default(),
                executor: self.executor,
                file_io_recv: self.io_recv,
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
    // The next worker ID.
    next_worker_id: AtomicUsize,

    // The global injector for scheduling tasks.
    scheduling_injector: Injector<Runnable<Meta>>,
    scheduling_stealers: RwLock<Vec<Stealer<Runnable<Meta>>>>,
    scheduling_workers: RwLock<Vec<Worker<Runnable<Meta>>>>,

    // The primary executor.
    executor: Arc<Executor<'static>>,

    // The I/O request queue.
    file_io_recv: Receiver<FileIoRequest>,
    /// The current count of I/O worker threads.
    active_io_workers: AtomicUsize,
    target_io_reqs_per_worker: usize, // e.g. queue len = 32, target = 8 -> 4 workers

    // The result channel.
    results: crossbeam_channel::Receiver<T>,
}

/// Metadata that we attach to each runnable task.
/// Currently, this is used exclusively to schedule tasks back onto the most recent worker that
/// polled them.
struct Meta {
    worker_affinity: Option<usize>,
}

impl<T: Send + 'static> WorkerPool<T> {
    pub fn new_worker(&self) -> Worker<T> {
        let id = self.shared.next_worker_id.fetch_add(1, Ordering::Relaxed);

        let io_runtime = tokio::runtime::Builder::new_current_thread()
            .thread_name(format!("vortex-worker-{}", id))
            .enable_io()
            .build()
            .vortex_expect("Failed to create worker I/O runtime");

        Worker {
            id,
            shared: self.shared.clone(),
            role: WorkerRole::Executor,
            io_runtime,
        }
    }
}

pub struct Worker<T: Send + 'static> {
    id: usize,
    shared: Arc<Shared<T>>,
    role: WorkerRole,

    // FIXME(ngates): we need to share a pool of workers that perform blocking reads...
    io_runtime: tokio::runtime::Runtime,
}

#[derive(Debug, Clone, Copy)]
enum WorkerRole {
    Executor,
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
        if self.shared.file_io_recv.is_empty() {
            // No work to do...
            return;
        }

        // We should become an I/O worker until there are no in-flight requests, and no requests
        // in the queue. After that, we yield back to the worker to perform more work.
        let file_io_recv = self.shared.file_io_recv.clone();
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
                        while let Some(fut) = inflight.next().await {}
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

            match self.update_role() {
                WorkerRole::Executor => {
                    // Otherwise, drive the main executor for some number of ticks.
                    block_on(self.shared.executor.run(async {
                        // We yield to allow other tasks to make progress.
                        smol::future::yield_now().await;
                    }));
                }
                WorkerRole::IO => {
                    self.drive_io();
                    // Start the loop again to check for more work.
                    continue;
                }
            }
        }
    }
}
