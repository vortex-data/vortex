// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{CpuTask, Handle, IoTask, Runtime};
use futures::executor::block_on;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::StreamExt;
use smol::Executor;
use std::num::NonZeroUsize;
use std::sync::Arc;
use vortex_error::VortexExpect;

pub struct MultiThreadRuntime<'rt> {
    executor: Arc<Executor<'rt>>,
    // Shutdown signal to stop all threads when dropped.
    _signal: kanal::Sender<()>,
}

impl<'rt> Runtime<'rt> for Executor<'rt> {
    fn spawn_scheduling(&self, fut: BoxFuture<'rt, ()>) {
        self.spawn(fut).detach();
    }

    fn spawn_cpu(&self, task: CpuTask) {
        self.spawn(async move { task.run() }).detach();
    }

    fn spawn_io(&self, stream: BoxStream<'rt, IoTask>, concurrency: usize) {
        self.spawn(async move {
            stream
                .map(|t: IoTask| t.run_send())
                .buffer_unordered(concurrency)
                .collect::<()>()
                .await
        })
        .detach();
    }
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
