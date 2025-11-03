// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::stream::BoxStream;
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use smol::block_on;
use smol::channel::{Sender, unbounded};
use vortex_error::VortexExpect;

use crate::runtime::{BlockingRuntime, Executor, Handle};

/// An async execution runtime that is broadly useful for driving Vortex scan and write operations.
///
/// This runtime is adaptive, upon construction it starts in current-thread mode, but can adaptively
/// be configured to execute multithreaded by setting the target number of worker threads.
///
/// The current thread runtime will do no work unless `block_on` is called. In other words, the
/// default behavior is single-threaded with code running on the thread that called `block_on`.
///
/// It's also possible to clone the runtime onto other threads, each of which can call `block_on`
/// to drive work on that thread. Each thread shares the same underlying executor with the same
/// set of tasks, allowing work to be driven in parallel.
///
/// Examples:
///
/// ```
/// # use std::sync::{Arc, Mutex};
/// # use smol::block_on;
/// # use vortex_io::runtime::DefaultRuntime;
/// # use vortex_io::runtime::BlockingRuntime;
/// # use std::collections::HashSet;
/// // Create a new runtime with current-thread execution.
/// let rt = DefaultRuntime::current_thread();
///
/// // We can block the runtime to force tasks to execute while blocking the current thread.
/// let value = rt.block_on(move |_| async move { "hello world".to_string() });
/// assert_eq!(value.as_str(), "hello world");
///
/// // We can adjust the number of worker threads, and spawn more tasks which will all be
/// // polled in parallel in the background.
/// rt.set_workers(2);
///
/// // Spawn some tasks.
/// let results = Arc::new(Mutex::new(HashSet::new()));
///
/// for id in 0..10 {
///     let result_clone = results.clone();
///     rt.handle().spawn(async move {
///         // Push a new result for the task.
///         result_clone.lock()
///             .unwrap()
///             .insert(format!("hello world - {id}"));
///     }).detach();
/// }
///
/// // The above tasks should complete, even though we didn't call block_on, because we're now
/// // operating with background workers.
/// loop {
///     let count = { results.lock().unwrap().len() };
///     if count == 10 {
///         break;
///     }
///     // yield so worker threads can do their thing.
///     std::thread::yield_now();
/// }
/// ```
#[derive(Clone, Default)]
pub struct DefaultRuntime {
    /// The underlying handle to an `async_executor::Executor`, which tracks all task queues
    /// that are concurrently being updated and stolen by worker threads.
    executor: Arc<smol::Executor<'static>>,
    /// A list of information necessary to clean up worker threads.
    workers: Arc<Mutex<Workers>>,
}

#[derive(Default)]
struct Workers {
    shutdown_handles: Vec<Sender<()>>,
}

impl DefaultRuntime {
    /// Create the runtime as a current-thread executor runtime.
    ///
    /// In this mode, futures spawned onto the runtime will not polled until the `block_on`
    /// method is called, at which point the thread which called `block_on` will poll any pending
    /// tasks necessary to resolve the future.
    pub fn current_thread() -> Self {
        Self::default()
    }

    /// Create a new runtime with multithreaded execution.
    ///
    /// In this mode, any futures spawned onto the runtime will be polled eagerly by `num_workers`
    /// background threads.
    ///
    /// A multithreaded runtime can be scaled back down to a [`current_thread`][Self::current_thread]
    /// runtime by calling `set_workers(0)`.
    pub fn multithread(num_workers: usize) -> Self {
        let this = Self::default();
        this.set_workers(num_workers);
        this
    }

    /// Update the number of workers available to run work.
    ///
    /// By default, this runtime will operate with current-thread execution, meaning all spawned
    /// futures will be polled on whichever thread calls the `block_on` method.
    ///
    /// By setting the number of workers here, callers can allow for background threads to eagerly
    /// poll several of the tasks. This means that we get access to some of these threads instead.
    pub fn set_workers(&self, num_workers: usize) {
        let mut workers = self.workers.lock();

        if num_workers < workers.shutdown_handles.len() {
            // Drop the shutdown handle for each extraneous runtime. This will cause the spawned
            // thread to complete as expected.
            workers.shutdown_handles.drain(num_workers..).for_each(drop);
        } else {
            let mut shutdown_signals =
                Vec::with_capacity(num_workers - workers.shutdown_handles.len());
            for worker_id in workers.shutdown_handles.len()..num_workers {
                let (signal, shutdown) = unbounded::<()>();
                let exec = self.executor.clone();
                std::thread::Builder::new()
                    .name(format!("vortex-runtime-worker-{}", worker_id))
                    .spawn(move || {
                        // NOTE: we explicitly discard the result of executing the recv future,
                        // because it will return an error when the sender end is closed, which is
                        // the normal shutdown flow for the runtime.
                        let _err = block_on(exec.run(shutdown.recv()));
                    })
                    .vortex_expect("spawning new worker threads should succeed on all platforms");

                // Push the shutdown handle only if the thread spawned successfully.
                // If any thread does not spawn successfully, we will unwind, which will drop `shutdown_signals`
                // and cause any intermediate spawned threads to die when their recv returns with error.
                shutdown_signals.push(signal);
            }

            // Only push the new shutdown handlers after all threads spawned successfully.
            workers.shutdown_handles.extend(shutdown_signals);
        }
    }

    /// Set the number of background worker threads based on the number of available cores,
    /// minus one to allow for a single driver core.
    pub fn set_workers_to_available_cores(&self) {
        let n = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1);
        self.set_workers(n);
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
                        log::trace!("all receivers dropped, stopping stream: {}", e);
                        break;
                    }
                }
            })
            .detach();

        ThreadSafeIterator {
            executor: self.executor.clone(),
            results: result_rx,
        }
    }
}

impl BlockingRuntime for DefaultRuntime {
    type BlockingIterator<'a, R: 'a> = BlockingIter<'a, R>;

    fn handle(&self) -> Handle {
        let executor: Arc<dyn Executor> = self.executor.clone();
        Handle::new(Arc::downgrade(&executor))
    }

    fn block_on<F, Fut, R>(&self, f: F) -> R
    where
        F: FnOnce(Handle) -> Fut,
        Fut: Future<Output = R>,
    {
        block_on(self.executor.run(f(self.handle())))
    }

    fn block_on_stream<'a, F, S, R>(&self, f: F) -> BlockingIter<'a, R>
    where
        F: FnOnce(Handle) -> S,
        S: Stream<Item = R> + Send + 'a,
        R: Send + 'a,
    {
        BlockingIter {
            executor: self.executor.clone(),
            stream: f(self.handle()).boxed(),
        }
    }
}

/// An iterator wrapping a stream, that calls `block_on` to fetch the next element by blocking
/// the calling thread.
pub struct BlockingIter<'a, T> {
    executor: Arc<smol::Executor<'static>>,
    stream: BoxStream<'a, T>,
}

impl<T> Iterator for BlockingIter<'_, T> {
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
            executor: self.executor.clone(),
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

#[allow(clippy::if_then_some_else_none)] // Clippy is wrong when if/else has await.
#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;

    use futures::{StreamExt, stream};
    use parking_lot::Mutex;

    use super::*;

    #[test]
    fn test_worker_thread() {
        let runtime = DefaultRuntime::current_thread();

        // We spawn a future that sets a value on a separate thread.
        let value = Arc::new(AtomicUsize::new(0));
        let value2 = value.clone();
        runtime
            .handle()
            .spawn(async move {
                value2.store(42, Ordering::SeqCst);
            })
            .detach();

        // By default, nothing has driven the executor, so the value should still be 0.
        assert_eq!(value.load(Ordering::SeqCst), 0);

        // An empty pool still does nothing.
        runtime.set_workers(0);
        assert_eq!(value.load(Ordering::SeqCst), 0);

        // Adding a worker thread should drive the executor.
        runtime.set_workers(1);

        // We need something to call block_on to make this shit work as expected.
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
        let mut iter = DefaultRuntime::current_thread()
            .block_on_stream(|_h| stream::iter(vec![1, 2, 3, 4, 5]).boxed());

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

        let iter = DefaultRuntime::current_thread()
            .block_on_stream_thread_safe(|_h| stream::iter(0..total_items).boxed());

        let barrier = Arc::new(Barrier::new(num_threads));
        let results = Arc::new(Mutex::new(Vec::new()));

        let threads: Vec<_> = (0..num_threads)
            .map(|_| {
                let mut iter = iter.clone();
                let counter = counter.clone();
                let barrier = barrier.clone();
                let results = results.clone();

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

        let iter = DefaultRuntime::current_thread().block_on_stream_thread_safe(|h| {
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
                let collected = collected.clone();
                let barrier = barrier.clone();

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
        let iter = DefaultRuntime::current_thread().block_on_stream(|h| {
            stream::unfold((h, 0), |(h, state)| async move {
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
        let c = counter.clone();

        let mut iter = DefaultRuntime::current_thread().block_on_stream(|_h| {
            stream::unfold(0, move |state| {
                let c = c.clone();
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
        let iter = DefaultRuntime::current_thread()
            .block_on_stream_thread_safe(|_h| stream::iter(0..20).boxed());

        let iter1 = iter.clone();
        let iter2 = iter;
        let barrier1 = barrier.clone();
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

        let iter = DefaultRuntime::current_thread()
            .block_on_stream_thread_safe(|_h| stream::iter(0..num_items).boxed());

        let received = Arc::new(Mutex::new(Vec::new()));
        let barrier = Arc::new(Barrier::new(num_threads));

        let threads: Vec<_> = (0..num_threads)
            .map(|_| {
                let iter = iter.clone();
                let received = received.clone();
                let barrier = barrier.clone();

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
