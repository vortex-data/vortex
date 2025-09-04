// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;

use futures::executor::block_on;
use futures::future::BoxFuture;
use smol::Executor;
use vortex_error::VortexExpect;

use crate::runtime::{AbortHandle, AbortHandleRef, Handle, Runtime};

/// A runtime that drives work in the background on a dedicated thread pool.
pub struct MultiThreadRuntime<'rt> {
    executor: Arc<Executor<'rt>>,
    // Shutdown signal to stop all threads when dropped.
    _signal: kanal::Sender<()>,
}

impl<'rt> MultiThreadRuntime<'rt> {
    pub fn new(workers: NonZeroUsize) -> Self {
        let executor = Arc::new(Executor::<'static>::new());

        let (signal, shutdown) = kanal::unbounded::<()>();

        for _ in 0..workers.get() {
            let executor = executor.clone();
            let shutdown = shutdown.clone();
            std::thread::Builder::new()
                .name("vortex-multi-thread".to_string())
                .spawn(move || {
                    block_on(executor.run(async move {
                        let _ = shutdown.as_async().recv().await;
                    }))
                })
                .vortex_expect("Cannot spawn multi-thread worker");
        }

        // Shorten the lifetime of the executor to tie it to the runtime. We need to do this
        // since the executor and runtime are self-referential.
        let executor =
            unsafe { std::mem::transmute::<Arc<Executor<'static>>, Arc<Executor<'rt>>>(executor) };

        MultiThreadRuntime {
            executor,
            _signal: signal,
        }
    }

    /// Drive the given Vortex future on the underlying multi-threaded runtime.
    pub fn drive<'fut, F, Fut, R>(&self, f: F) -> R
    where
        F: FnOnce(Handle<'rt>) -> Fut,
        Fut: Future<Output = R> + 'fut,
        R: Send + 'static,
    {
        let fut = f(Handle(self.executor.clone()));
        block_on(self.executor.run(fut))
    }
}

impl Default for MultiThreadRuntime<'_> {
    fn default() -> Self {
        Self::new(std::thread::available_parallelism().unwrap_or_else(|_| NonZeroUsize::MIN))
    }
}

impl<'rt> Runtime<'rt> for Executor<'rt> {
    fn spawn(&self, fut: BoxFuture<'rt, ()>) -> AbortHandleRef<'rt> {
        SmolAbortHandle::new(self.spawn(fut))
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef<'rt> {
        // For now, we spawn CPU work back onto the same executor.
        SmolAbortHandle::new(self.spawn(async move { task() }))
    }
}

/// An abort handle for a `smol::Task`.
pub(crate) struct SmolAbortHandle<T> {
    task: Option<smol::Task<T>>,
}

impl<'rt, T: 'rt + Send> SmolAbortHandle<T> {
    pub(crate) fn new(task: smol::Task<T>) -> AbortHandleRef<'rt> {
        Box::new(Self { task: Some(task) })
    }
}

impl<T: Send> AbortHandle<'_> for SmolAbortHandle<T> {
    fn abort(mut self: Box<Self>) {
        // Aborting a smol::Task is done by dropping it.
        drop(self.task.take());
    }
}

impl<T> Drop for SmolAbortHandle<T> {
    fn drop(&mut self) {
        // We prevent the task from being canceled by detaching it.
        if let Some(task) = self.task.take() {
            task.detach()
        }
    }
}
