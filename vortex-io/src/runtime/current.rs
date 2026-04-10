// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::Stream;
use futures::StreamExt;
use futures::stream::BoxStream;
use smol::block_on;

use crate::runtime::BlockingRuntime;
use crate::runtime::Executor;
use crate::runtime::Handle;
pub use crate::runtime::pool::CurrentThreadWorkerPool;

/// A current thread runtime allows callers to much more explicitly drive Vortex futures than with
/// a Tokio runtime.
///
/// The current thread runtime will do no work unless `block_on` is called. In other words, the
/// default behavior is single-threaded with code running on the thread that called `block_on`.
///
/// It's also possible to clone the runtime onto other threads, each of which can call `block_on`
/// to drive work on that thread. Each thread shares the same underlying executor with the same
/// set of tasks, allowing work to be driven in parallel.
///
/// For automatic driving of work, a [`CurrentThreadWorkerPool`] can be created from the runtime
/// by calling [`new_pool`](CurrentThreadRuntime::new_pool). The returned pool can be configured
/// with the desired number of worker threads that will drive work on behalf of the runtime.
#[derive(Clone, Default)]
pub struct CurrentThreadRuntime {
    executor: Arc<smol::Executor<'static>>,
}

impl CurrentThreadRuntime {
    /// Create a new current thread runtime.
    pub fn new() -> Self {
        Self::default()
    }

    /// Create a new worker pool for driving the runtime in the background.
    ///
    /// This pool can be used to offload work from the current thread to a set of worker threads
    /// that will drive the runtime's executor.
    ///
    /// By default, the pool has no worker threads; the caller must set the desired number of
    /// worker threads using the `set_workers` method on the returned pool.
    pub fn new_pool(&self) -> CurrentThreadWorkerPool {
        CurrentThreadWorkerPool::new(Arc::clone(&self.executor))
    }

    /// Returns an iterator wrapper around a stream, blocking the current thread for each item.
    ///
    /// ## Multi-threaded Usage
    ///
    /// To drive the iterator from multiple threads, simply clone it and call `next()` on each
    /// clone. Results on each thread are ordered with respect to the stream, but there is no
    /// ordering guarantee between threads.
    pub fn block_on_stream_thread_safe<F, S, R>(&self, f: F) -> ThreadSafeIterator<R>
    where
        F: FnOnce(Handle) -> S,
        S: Stream<Item = R> + Send + 'static,
        R: Send + 'static,
    {
        let stream = f(self.handle());

        // We create an MPMC result channel and spawn a task to drive the stream and send results.
        // This allows multiple worker threads to drive the execution while all waiting for results
        // on the channel.
        let (result_tx, result_rx) = kanal::bounded_async(1);
        self.executor
            .spawn(async move {
                futures::pin_mut!(stream);
                while let Some(item) = stream.next().await {
                    // If all receivers are dropped, we stop driving the stream.
                    if let Err(e) = result_tx.send(item).await {
                        tracing::trace!("all receivers dropped, stopping stream: {}", e);
                        break;
                    }
                }
            })
            .detach();

        ThreadSafeIterator {
            executor: Arc::clone(&self.executor),
            results: result_rx,
        }
    }
}

impl BlockingRuntime for CurrentThreadRuntime {
    type BlockingIterator<'a, R: 'a> = CurrentThreadIterator<'a, R>;

    fn handle(&self) -> Handle {
        let executor: Arc<dyn Executor> = Arc::clone(&self.executor) as Arc<dyn Executor>;
        Handle::new(Arc::downgrade(&executor))
    }

    fn block_on<Fut, R>(&self, fut: Fut) -> R
    where
        Fut: Future<Output = R>,
    {
        block_on(self.executor.run(fut))
    }

    fn block_on_stream<'a, S, R>(&self, stream: S) -> Self::BlockingIterator<'a, R>
    where
        S: Stream<Item = R> + Send + 'a,
        R: Send + 'a,
    {
        CurrentThreadIterator {
            executor: Arc::clone(&self.executor),
            stream: stream.boxed(),
        }
    }
}

/// An iterator that wraps up a stream to drive it using the current thread execution.
pub struct CurrentThreadIterator<'a, T> {
    executor: Arc<smol::Executor<'static>>,
    stream: BoxStream<'a, T>,
}

impl<T> Iterator for CurrentThreadIterator<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        block_on(self.executor.run(self.stream.next()))
    }
}

/// An iterator that drives a stream from multiple threads.
pub struct ThreadSafeIterator<T> {
    executor: Arc<smol::Executor<'static>>,
    results: kanal::AsyncReceiver<T>,
}

// Manual clone implementation since `T` does not need to be `Clone`.
impl<T> Clone for ThreadSafeIterator<T> {
    fn clone(&self) -> Self {
        Self {
            executor: Arc::clone(&self.executor),
            results: self.results.clone(),
        }
    }
}

impl<T> Iterator for ThreadSafeIterator<T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        block_on(self.executor.run(self.results.recv())).ok()
    }
}

#[expect(clippy::if_then_some_else_none)] // Clippy is wrong when if/else has await.
#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::Barrier;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::thread;
    use std::time::Duration;

    use futures::StreamExt;
    use futures::stream;
    use parking_lot::Mutex;

    use super::*;

    #[test]
    fn test_worker_thread() {
        let runtime = CurrentThreadRuntime::new();

        // We spawn a future that sets a value on a separate thread.
        let value = Arc::new(AtomicUsize::new(0));
        let value2 = Arc::clone(&value);
        runtime
            .handle()
            .spawn(async move {
                value2.store(42, Ordering::SeqCst);
            })
            .detach();

        // By default, nothing has driven the executor, so the value should still be 0.
        assert_eq!(value.load(Ordering::SeqCst), 0);

        // An empty pool still does nothing.
        let pool = runtime.new_pool();
        assert_eq!(value.load(Ordering::SeqCst), 0);

        // Adding a worker thread should drive the executor.
        pool.set_workers(1);
        for _ in 0..10 {
            if value.load(Ordering::SeqCst) == 42 {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        assert_eq!(value.load(Ordering::SeqCst), 42);
    }

    #[test]
    fn test_block_on_stream_single_thread() {
        let mut iter =
            CurrentThreadRuntime::new().block_on_stream(stream::iter(vec![1, 2, 3, 4, 5]).boxed());

        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), Some(4));
        assert_eq!(iter.next(), Some(5));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn test_block_on_stream_multiple_threads() {
        let counter = Arc::new(AtomicUsize::new(0));
        let num_threads = 4;
        let items_per_thread = 25;
        let total_items = 100;

        let iter = CurrentThreadRuntime::new()
            .block_on_stream_thread_safe(|_h| stream::iter(0..total_items).boxed());

        let barrier = Arc::new(Barrier::new(num_threads));
        let results = Arc::new(Mutex::new(Vec::new()));

        let threads: Vec<_> = (0..num_threads)
            .map(|_| {
                let mut iter = iter.clone();
                let counter = Arc::clone(&counter);
                let barrier = Arc::clone(&barrier);
                let results = Arc::clone(&results);

                thread::spawn(move || {
                    barrier.wait();
                    let mut local_results = Vec::new();

                    for _ in 0..items_per_thread {
                        if let Some(item) = iter.next() {
                            counter.fetch_add(1, Ordering::SeqCst);
                            local_results.push(item);
                        }
                    }

                    results.lock().push(local_results);
                })
            })
            .collect();

        for thread in threads {
            thread.join().unwrap();
        }

        assert_eq!(counter.load(Ordering::SeqCst), total_items);

        let all_results = results.lock();
        let mut collected: Vec<_> = all_results.iter().flatten().copied().collect();
        collected.sort();
        assert_eq!(collected, (0..total_items).collect::<Vec<_>>());
    }

    #[test]
    fn test_block_on_stream_concurrent_clone_and_drive() {
        let num_items = 50;
        let num_threads = 3;

        let iter = CurrentThreadRuntime::new().block_on_stream_thread_safe(|h| {
            stream::unfold(0, move |state| {
                let h = h.clone();
                async move {
                    if state < num_items {
                        h.spawn_cpu(move || {
                            thread::sleep(Duration::from_micros(10));
                            state
                        })
                        .await;
                        Some((state, state + 1))
                    } else {
                        None
                    }
                }
            })
        });

        let collected = Arc::new(Mutex::new(Vec::new()));
        let barrier = Arc::new(Barrier::new(num_threads));

        let threads: Vec<_> = (0..num_threads)
            .map(|thread_id| {
                let iter = iter.clone();
                let collected = Arc::clone(&collected);
                let barrier = Arc::clone(&barrier);

                thread::spawn(move || {
                    barrier.wait();
                    let mut local_items = Vec::new();

                    for item in iter {
                        local_items.push((thread_id, item));
                        if local_items.len() >= 5 {
                            break;
                        }
                    }

                    collected.lock().extend(local_items);
                })
            })
            .collect();

        for thread in threads {
            thread.join().unwrap();
        }

        let results = collected.lock();
        let mut values: Vec<_> = results.iter().map(|(_, v)| *v).collect();
        values.sort();
        values.dedup();

        assert!(values.len() >= 5);
        assert!(values.iter().all(|&v| v < num_items));
    }

    #[test]
    fn test_block_on_stream_async_work() {
        let runtime = CurrentThreadRuntime::new();
        let handle = runtime.handle();
        let iter = runtime.block_on_stream({
            stream::unfold((handle, 0), |(h, state)| async move {
                if state < 10 {
                    let value = h
                        .spawn(async move { futures::future::ready(state * 2).await })
                        .await;
                    Some((value, (h, state + 1)))
                } else {
                    None
                }
            })
        });

        let results: Vec<_> = iter.collect();
        assert_eq!(results, vec![0, 2, 4, 6, 8, 10, 12, 14, 16, 18]);
    }

    #[test]
    fn test_block_on_stream_drop_receivers_early() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);

        let mut iter = CurrentThreadRuntime::new().block_on_stream({
            stream::unfold(0, move |state| {
                let c = Arc::clone(&c);
                async move {
                    (state < 100).then(|| {
                        c.fetch_add(1, Ordering::SeqCst);
                        (state, state + 1)
                    })
                }
            })
            .boxed()
        });

        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(2));

        drop(iter);

        let final_count = counter.load(Ordering::SeqCst);
        assert!(
            final_count < 100,
            "Stream should stop when all receivers are dropped"
        );
    }

    #[test]
    fn test_block_on_stream_interleaved_access() {
        let barrier = Arc::new(Barrier::new(2));
        let iter = CurrentThreadRuntime::new()
            .block_on_stream_thread_safe(|_h| stream::iter(0..20).boxed());

        let iter1 = iter.clone();
        let iter2 = iter;
        let barrier1 = Arc::clone(&barrier);
        let barrier2 = barrier;

        let thread1 = thread::spawn(move || {
            let mut iter = iter1;
            let mut results = Vec::new();
            barrier1.wait();

            for _ in 0..5 {
                if let Some(val) = iter.next() {
                    results.push(val);
                    thread::sleep(Duration::from_micros(50));
                }
            }
            results
        });

        let thread2 = thread::spawn(move || {
            let mut iter = iter2;
            let mut results = Vec::new();
            barrier2.wait();

            for _ in 0..5 {
                if let Some(val) = iter.next() {
                    results.push(val);
                    thread::sleep(Duration::from_micros(50));
                }
            }
            results
        });

        let results1 = thread1.join().unwrap();
        let results2 = thread2.join().unwrap();

        let mut all_results = results1;
        all_results.extend(results2);
        all_results.sort();

        assert_eq!(all_results, (0..10).collect::<Vec<_>>());

        for i in 0..10 {
            assert_eq!(all_results.iter().filter(|&&x| x == i).count(), 1);
        }
    }

    #[test]
    fn test_block_on_stream_stress_test() {
        let num_threads = 10;
        let num_items = 1000;

        let iter = CurrentThreadRuntime::new()
            .block_on_stream_thread_safe(|_h| stream::iter(0..num_items).boxed());

        let received = Arc::new(Mutex::new(Vec::new()));
        let barrier = Arc::new(Barrier::new(num_threads));

        let threads: Vec<_> = (0..num_threads)
            .map(|_| {
                let iter = iter.clone();
                let received = Arc::clone(&received);
                let barrier = Arc::clone(&barrier);

                thread::spawn(move || {
                    barrier.wait();
                    for val in iter {
                        received.lock().push(val);
                    }
                })
            })
            .collect();

        for thread in threads {
            thread.join().unwrap();
        }

        let mut results = received.lock().clone();
        results.sort();

        assert_eq!(results.len(), num_items);
        assert_eq!(results, (0..num_items).collect::<Vec<_>>());
    }
}
