// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{FileIoRequest, Runtime};
use flume::Receiver;
use futures::{pin_mut, Stream};
use futures_util::stream::FuturesUnordered;
use futures_util::task::noop_waker_ref;
use futures_util::StreamExt;
use smol::lock::Semaphore;
use smol::Executor;
use std::os::unix::fs::FileExt;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::task::{Context, Poll};
use tokio::task::{spawn_blocking, LocalSet};
use vortex_buffer::ByteBufferMut;
use vortex_error::{vortex_err, VortexError, VortexExpect, VortexResult};

impl Runtime {
    /// Returns a worker pool that can be used to drive the Runtime and in the process emit
    /// items from the stream.
    pub fn drive_stream_on_pool<T: Send + 'static>(
        self,
        stream: impl Stream<Item = T> + Send + 'static,
    ) -> WorkerPool<T> {
        // We create a channel for the output results and spawn a detached task to populate it.
        let (send, recv) = flume::unbounded::<T>();
        self.executor
            .spawn(async move {
                pin_mut!(stream);
                while let Some(item) = stream.next().await {
                    if let Err(e) = send.send(item) {
                        log::trace!("Failed to send item to worker pool: {}", e);
                        break;
                    }
                }
            })
            .detach();

        WorkerPool {
            shared: Arc::new(Shared {
                next_worker_id: Default::default(),
                executor: self.executor,
                file_io_recv: self.file_io_recv,
                io_workers: Arc::new(Semaphore::new(1)), // Start with 1 I/O worker.
                results: recv,
            }),
        }
    }
}

pub struct WorkerPool<T: Send + 'static> {
    shared: Arc<Shared<T>>,
}

struct Shared<T: Send + 'static> {
    // The next worker ID.
    next_worker_id: AtomicUsize,

    // The primary executor.
    executor: Arc<Executor<'static>>,

    // The I/O request queue.
    file_io_recv: Receiver<FileIoRequest>,
    // A semaphore for controlling the number of I/O workers.
    io_workers: Arc<Semaphore>,

    // The result channel.
    results: Receiver<T>,
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
            shared: self.shared.clone(),
            io_runtime,
        }
    }
}

pub struct Worker<T: Send + 'static> {
    shared: Arc<Shared<T>>,

    // FIXME(ngates): we need to share a pool of workers that perform blocking reads...
    io_runtime: tokio::runtime::Runtime,
}

impl<T: Send + 'static> Worker<T> {
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
                // Create a FuturesUnordered to manage concurrent blocking operations
                let mut inflight = FuturesUnordered::<VortexResult<()>>::new();

                // Convert receiver to stream for easier polling
                let mut file_io_stream = file_io_recv.into_stream();

                loop {
                    // Try to fill up to our concurrency limit with new requests
                    let mut got_new_request = false;

                    while inflight.len() < 16 {
                        let mut cx = Context::from_waker(noop_waker_ref());

                        match file_io_stream.poll_next_unpin(&mut cx) {
                            Poll::Ready(Some(req)) => {
                                got_new_request = true;

                                // Spawn a new blocking operation
                                // let fut = async move {
                                // spawn_blocking(move || {
                                let mut buffer =
                                    ByteBufferMut::with_capacity_aligned(req.length, req.alignment);
                                unsafe { buffer.set_len(req.length) };
                                match req.file.read_exact_at(&mut buffer, req.offset) {
                                    Ok(()) => req.resolve(Ok(buffer.freeze())),
                                    Err(e) => req.resolve(Err(VortexError::from(e))),
                                }
                                // })
                                // .await
                                // .map_err(|e| {
                                //     vortex_err!("Failed to spawn blocking read: {}", e)
                                // })
                                // };
                                //
                                // inflight.push(fut);
                            }
                            Poll::Ready(None) => {
                                // Channel closed
                                // Wait for remaining operations to complete before terminating
                                while let Some(result) = inflight.next().await {
                                    result.vortex_expect("Failed to complete blocking read");
                                }
                                return;
                            }
                            Poll::Pending => {
                                // No more requests available right now
                                break;
                            }
                        }
                    }

                    // If we have pending operations, wait for at least one to complete
                    if !inflight.is_empty() {
                        if let Some(result) = inflight.next().await {
                            result.vortex_expect("Failed to complete blocking read");
                        }
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

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Try to emit an item from the results channel.
            match self.shared.results.try_recv() {
                Ok(item) => return Some(item),
                Err(flume::TryRecvError::Empty) => { /* No items, continue */ }
                Err(flume::TryRecvError::Disconnected) => return None,
            }

            // Check if we should become an I/O worker.
            if let Some(_guard) = self.shared.io_workers.try_acquire_arc() {
                if !self.shared.file_io_recv.is_empty() {
                    self.drive_io();
                    // Start the loop again to check for more work.
                    continue;
                }
            }

            // Otherwise, drive the main executor for some number of ticks.
            // FIXME(ngates): adjust based on load?
            for _ in 0..8 {
                if !self.shared.executor.try_tick() {
                    break;
                }
            }
        }
    }
}
