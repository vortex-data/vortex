// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;

use futures::stream::BoxStream;
use futures::{Stream, StreamExt};
use smol::{Executor, block_on};
use vortex_error::VortexExpect;

use crate::runtime::Handle;

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

    /// Drive the given Vortex future on the underlying multithreaded runtime.
    pub fn block_on<'scope, F, Fut, R>(&self, f: F) -> R
    where
        F: FnOnce(Handle<'scope, 'rt>) -> Fut,
        Fut: Future<Output = R> + 'scope,
        R: Send + 'scope,
        'rt: 'scope,
    {
        let fut = f(Handle::new(self.executor.clone()));
        block_on(self.executor.run(fut))
    }

    /// Drive the given Vortex stream on the underlying multithreaded runtime.
    pub fn block_on_stream<'scope, F, S, R>(&self, f: F) -> impl Iterator<Item = R> + 'scope
    where
        F: FnOnce(Handle<'scope, 'rt>) -> S,
        S: Stream<Item = R> + Send + Unpin + 'scope,
        R: Send + 'scope,
        'rt: 'scope,
    {
        let stream = f(Handle::new(self.executor.clone()));
        BlockingStream { executor: self.executor.clone(), stream: stream.boxed() }
    }
}

impl Default for MultiThreadRuntime<'_> {
    fn default() -> Self {
        Self::new(std::thread::available_parallelism().unwrap_or(NonZeroUsize::MIN))
    }
}

/// A stream that wraps up the stream with the executor that drives it.
///
/// This allows the resulting stream to have a static lifetime.
struct BlockingStream<'scope, 'rt, T> {
    executor: Arc<Executor<'rt>>,
    stream: BoxStream<'scope, T>,
}

impl<T> Iterator for BlockingStream<'_, '_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let fut = self.stream.next();
        block_on(self.executor.run(fut))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::MultiThreadRuntime;

    #[test]
    fn test_drive_simple_future() {
        let rt = MultiThreadRuntime::new(NonZeroUsize::new(2).unwrap());
        let result = rt.block_on(|_| async { 42 });
        assert_eq!(result, 42);
    }
}
