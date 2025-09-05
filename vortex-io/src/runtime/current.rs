// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::{Stream, StreamExt};
use smol::{Executor, block_on};

use crate::runtime::Handle;

/// A current thread runtime allows users to explicitly drive Vortex futures from multiple worker
/// threads that they manage. This is useful in environments where the user already has a thread
/// pool and wants to integrate Vortex into that pool, for example query engines.
pub struct CurrentThreadRuntime;

impl CurrentThreadRuntime {
    /// Drive the given Vortex future on the underlying current thread runtime.
    pub fn block_on<'rt, F, Fut, R>(f: F) -> R
    where
        F: FnOnce(Handle<'rt>) -> Fut,
        Fut: Future<Output = R> + 'rt,
        R: Send + 'static,
    {
        let executor = Arc::new(Executor::new());
        let fut = f(Handle(executor.clone()));
        block_on(executor.run(fut))
    }

    /// Drive the given Vortex stream on the underlying current thread runtime.
    ///
    /// Note the resulting [`Iterator`] supports [`Clone`] in order to drive the stream from
    /// multiple threads.
    pub fn block_on_stream<'rt, F, S, R>(f: F) -> impl Iterator<Item = R> + Clone
    where
        F: FnOnce(Handle<'rt>) -> S,
        S: Stream<Item = R> + Send + Unpin + 'rt,
        R: Send + 'rt,
    {
        let executor = Arc::new(Executor::new());
        let stream = f(Handle(executor.clone()));

        // We create an MPMC result channel and spawn a task to drive the stream and send results.
        // This allows multiple worker threads to drive the executor while all waiting for results
        // on the channel.
        let (result_tx, result_rx) = kanal::unbounded();
        executor
            .spawn(async move {
                futures::pin_mut!(stream);
                while let Some(item) = stream.next().await {
                    // Ignore send errors, which happen if all receivers are dropped.
                    let _ = result_tx.send(item);
                }
            })
            .detach();

        // SAFETY: the returned stream is self-referential and lives as long as the executor.
        //  We therefore extend the lifetime of the executor to 'static.
        let executor: Arc<Executor<'static>> =
            unsafe { std::mem::transmute::<Arc<Executor<'_>>, Arc<Executor<'static>>>(executor) };

        BlockingStream {
            executor,
            results: result_rx.to_async(),
        }
    }
}

/// A stream that wraps up the stream with the executor that drives it.
///
/// This allows the resulting stream to have a static lifetime.
struct BlockingStream<T> {
    executor: Arc<Executor<'static>>,
    results: kanal::AsyncReceiver<T>,
}

// Manually implement Clone since `T` doesn't need to be `Clone`.
impl<T> Clone for BlockingStream<T> {
    fn clone(&self) -> Self {
        BlockingStream {
            executor: self.executor.clone(),
            results: self.results.clone(),
        }
    }
}

impl<T> Iterator for BlockingStream<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        block_on(self.executor.run(self.results.recv())).ok()
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use super::*;

    #[test]
    fn test_drive_simple_future() {
        let result = CurrentThreadRuntime::block_on(|_handle| async { 123 });
        assert_eq!(result, 123);
    }

    #[test]
    fn test_spawn_cpu_task() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();

        let result = CurrentThreadRuntime::block_on(|handle| async move {
            handle
                .spawn_cpu(move || c.fetch_add(1, Ordering::SeqCst))
                .await
        });

        assert_eq!(result, 0);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
