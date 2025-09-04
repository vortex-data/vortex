// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::executor::block_on;
use futures::future::BoxFuture;
use futures::{FutureExt, Stream, StreamExt};
use smol::Executor;

use crate::runtime::multithread::SmolAbortHandle;
use crate::runtime::{AbortHandleRef, Handle, Runtime};

/// A worker pool is a Vortex runtime that can be driven from multiple worker threads, typically
/// orchestrated by the system that is calling into Vortex.
///
/// Each worker makes a decision about whether to perform I/O tasks, CPU tasks, or drive the
/// underlying stream. It is therefore expected that the stream is largely a lightweight state
/// machine that alternates between spawning I/O and spawning CPU onto the runtime handle.
pub struct WorkerPool<T: Send> {
    shared: Arc<Shared<'static, T>>,
}

impl<T: Send> WorkerPool<T> {
    pub fn drive_stream<'rt, F, S>(f: F) -> WorkerPool<T>
    where
        F: FnOnce(Handle<'rt>) -> S,
        S: Stream<Item = T> + Send + 'rt,
        T: 'rt,
    {
        let (result_tx, result_rx) = kanal::unbounded();

        let shared = Arc::new(Shared {
            executor: Arc::new(Executor::<'rt>::new()),
            results: result_rx,
        });

        let handle = Handle(shared.clone());
        let stream = f(handle.clone());

        // Spawn a task to drive the stream and send results to the result channel.
        shared.spawn(
            async move {
                futures::pin_mut!(stream);
                while let Some(item) = stream.next().await {
                    // Ignore send errors, which happen if the receiver is dropped.
                    let _ = result_tx.send(item);
                }
            }
            .boxed(),
        );

        let shared =
            unsafe { std::mem::transmute::<Arc<Shared<'rt, T>>, Arc<Shared<'static, T>>>(shared) };

        WorkerPool { shared }
    }
}

struct Shared<'rt, T: Send> {
    executor: Arc<Executor<'rt>>,
    // The result channel.
    results: kanal::Receiver<T>,
}

/// We implement [`Runtime`] for the worker pool's shared state, which allows us to create a handle
/// that spawns onto the shared injector queues.
///
/// Note that we _also_ implement [`Runtime`] for each individual worker, which allows us to pass
/// a handle that spawns onto a specific worker's local queues.
impl<'rt, T: Send> Runtime<'rt> for Shared<'rt, T> {
    fn spawn(&self, fut: BoxFuture<'rt, ()>) -> AbortHandleRef<'rt> {
        SmolAbortHandle::new(self.executor.spawn(fut))
    }

    fn spawn_cpu(&self, cpu: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef<'rt> {
        SmolAbortHandle::new(self.executor.spawn(async move { cpu() }))
    }
}

impl<T: Send + 'static> WorkerPool<T> {
    pub fn new_worker(&self) -> Worker<T> {
        Worker {
            shared: self.shared.clone(),
        }
    }
}

pub struct Worker<T: Send + 'static> {
    shared: Arc<Shared<'static, T>>,
}

/// Implementation of an iterator that actually drives the underlying runtime.
impl<T: Send + 'static> Iterator for Worker<T> {
    type Item = T;

    #[inline(never)]
    fn next(&mut self) -> Option<Self::Item> {
        // Run the executor until we get a result from the channel.
        block_on(
            self.shared
                .executor
                .run(self.shared.results.as_async().recv()),
        )
        .ok()
    }
}
