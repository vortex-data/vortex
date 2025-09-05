// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;
use std::sync::Arc;

use futures::Stream;
use futures_util::StreamExt;
use futures_util::stream::BoxStream;
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
    pub fn block_on<'fut, F, Fut, R>(&self, f: F) -> R
    where
        F: FnOnce(Handle<'rt>) -> Fut,
        Fut: Future<Output = R> + 'fut,
        R: Send + 'static,
    {
        let fut = f(Handle(self.executor.clone()));
        block_on(self.executor.run(fut))
    }

    /// Drive the given Vortex stream on the underlying multithreaded runtime.
    pub fn block_on_stream<F, S, R>(&self, f: F) -> impl Iterator<Item = R>
    where
        F: FnOnce(Handle<'rt>) -> S,
        S: Stream<Item = R> + Send + Unpin,
        R: Send + 'static,
    {
        let stream = f(Handle(self.executor.clone()));

        // SAFETY: The stream contains references to `rt` with lifetime 'rt.
        // We're transmuting this to 'static, which is sound because:
        // 1. Both `rt` and `stream` will be moved into BlockingStream
        // 2. BlockingStream will drop them in the correct order (stream first, then rt)
        // 3. The stream will never outlive the runtime it references
        let stream: BoxStream<'static, R> = unsafe {
            std::mem::transmute::<BoxStream<'_, R>, BoxStream<'static, R>>(stream.boxed())
        };
        let executor: Arc<Executor<'static>> = unsafe {
            std::mem::transmute::<Arc<Executor<'_>>, Arc<Executor<'static>>>(self.executor.clone())
        };

        BlockingStream { executor, stream }
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
struct BlockingStream<T> {
    executor: Arc<Executor<'static>>,
    stream: BoxStream<'static, T>,
}

impl<T> Iterator for BlockingStream<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let fut = self.stream.next();
        block_on(self.executor.run(fut))
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use super::*;

    #[test]
    fn test_drive_simple_future() {
        let rt = MultiThreadRuntime::new(NonZeroUsize::new(2).unwrap());
        let result = rt.block_on(|_| async { 42 });
        assert_eq!(result, 42);
    }
}
