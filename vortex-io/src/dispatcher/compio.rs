// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::future::Future;
use std::panic::resume_unwind;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::thread::JoinHandle;

use compio::runtime::{JoinHandle as CompioJoinHandle, Runtime, RuntimeBuilder};
use futures::channel::oneshot;
use tracing::Instrument;
use vortex_error::{VortexResult, vortex_bail, vortex_panic};

use super::{Dispatch, JoinHandle as VortexJoinHandle};

trait CompioSpawn {
    fn spawn(self: Box<Self>) -> CompioJoinHandle<()>;
}

struct CompioTask<F, R> {
    task: F,
    result: oneshot::Sender<R>,
    span: tracing::Span,
}

impl<F, Fut, R> CompioSpawn for CompioTask<F, R>
where
    F: FnOnce() -> Fut + Send + 'static,
    Fut: Future<Output = R>,
    R: Send + 'static,
{
    fn spawn(self: Box<Self>) -> CompioJoinHandle<()> {
        let CompioTask { task, result, span } = *self;
        Runtime::with_current(|rt| {
            rt.spawn(async move {
                let task_output = task().instrument(span).await;
                result.send(task_output).ok();
            })
        })
    }
}

#[derive(Debug)]
pub(super) struct CompioDispatcher {
    submitter: kanal::Sender<Box<dyn CompioSpawn + Send>>,
    threads: Vec<JoinHandle<()>>,
    shutdown_flag: Arc<AtomicBool>,
}

impl CompioDispatcher {
    pub fn new(num_threads: usize) -> Self {
        let (submitter, rx) = kanal::unbounded();
        let rx = rx.to_async();
        let shutdown_flag = Arc::new(AtomicBool::new(false));
        let threads: Vec<_> = (0..num_threads)
            .map(|tid| {
                let worker_thread = std::thread::Builder::new();
                let worker_thread = worker_thread.name(format!("compio-dispatch-{tid}"));
                let rx: kanal::AsyncReceiver<Box<dyn CompioSpawn + Send>> = rx.clone();
                let shutdown = shutdown_flag.clone();

                worker_thread
                    .spawn(move || {
                        // Create a runtime-per-thread
                        let rt = RuntimeBuilder::new().build().unwrap_or_else(|e| {
                            vortex_panic!("CompioDispatcher RuntimeBuilder build(): {e}")
                        });

                        rt.block_on(async move {
                            // Use try_recv_async with timeout to periodically check shutdown flag
                            loop {
                                // Check shutdown flag
                                if shutdown.load(Ordering::Relaxed) {
                                    break;
                                }

                                // Try to receive with a timeout
                                match rx.recv().await {
                                    Ok(task) => task.spawn().detach(),
                                    Err(_) => {
                                        // Channel closed, exit gracefully
                                        break;
                                    }
                                }
                            }
                        });
                    })
                    .unwrap_or_else(|e| vortex_panic!("CompioDispatcher worker thread spawn: {e}"))
            })
            .collect();

        Self {
            submitter,
            threads,
            shutdown_flag,
        }
    }
}

impl Dispatch for CompioDispatcher {
    fn dispatch<F, Fut, R>(&self, task: F) -> VortexResult<VortexJoinHandle<R>>
    where
        F: (FnOnce() -> Fut) + Send + 'static,
        Fut: Future<Output = R> + 'static,
        R: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        let compio_task = Box::new(CompioTask {
            task,
            result: tx,
            span: tracing::Span::current(),
        });
        match self.submitter.send(compio_task) {
            Ok(()) => Ok(VortexJoinHandle(rx)),
            Err(err) => vortex_bail!("Dispatcher error spawning task: {err}"),
        }
    }

    fn shutdown(self) -> VortexResult<()> {
        // Signal shutdown to all threads
        self.shutdown_flag.store(true, Ordering::Relaxed);

        // drop the submitter.
        // Each worker thread will receive an `Err(Canceled)`
        drop(self.submitter);

        // Wait for all threads to finish their current work and exit.
        // This will wait indefinitely, but that's correct - we want to ensure
        // all work completes, even if it's a long-running operation.
        for thread in self.threads {
            match thread.join() {
                Ok(()) => {}
                Err(err) => {
                    // If a thread panicked, propagate the panic
                    resume_unwind(err);
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
#[cfg(feature = "compio")]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    use vortex_error::VortexResult;

    use super::*;

    // Helper function to wait for dispatcher results in tests
    // Simple busy-wait for oneshot receivers which complete quickly
    fn wait_for<R>(handle: VortexJoinHandle<R>) -> VortexResult<R> {
        use std::task::{Context, Poll};

        use futures::FutureExt;

        let mut handle = Box::pin(handle);
        loop {
            // Create a no-op waker that does nothing when woken
            let waker = futures::task::noop_waker();
            let mut cx = Context::from_waker(&waker);

            match handle.poll_unpin(&mut cx) {
                Poll::Ready(result) => return result,
                Poll::Pending => std::thread::yield_now(),
            }
        }
    }

    #[test]
    fn test_dispatcher_creation() {
        let dispatcher = CompioDispatcher::new(1);
        assert_eq!(dispatcher.threads.len(), 1);
        dispatcher.shutdown().unwrap();
    }

    #[test]
    fn test_dispatcher_single_thread() {
        let dispatcher = CompioDispatcher::new(1);
        assert_eq!(dispatcher.threads.len(), 1);
        dispatcher.shutdown().unwrap();
    }

    #[test]
    fn test_dispatch_simple_task() {
        let dispatcher = CompioDispatcher::new(1);

        let handle = dispatcher.dispatch(|| async { 42 }).unwrap();

        let result = wait_for(handle);
        assert_eq!(result.unwrap(), 42);

        dispatcher.shutdown().unwrap();
    }

    #[test]
    fn test_dispatch_multiple_tasks() {
        let dispatcher = CompioDispatcher::new(2);
        let mut handles = Vec::new();

        for i in 0..10 {
            let handle = dispatcher.dispatch(move || async move { i * 2 }).unwrap();
            handles.push(handle);
        }

        for (i, handle) in handles.into_iter().enumerate() {
            let result = wait_for(handle);
            assert_eq!(result.unwrap(), i * 2);
        }

        dispatcher.shutdown().unwrap();
    }

    #[test]
    fn test_concurrent_task_execution() {
        let dispatcher = CompioDispatcher::new(2);
        let counter = Arc::new(AtomicUsize::new(0));
        let mut handles = Vec::new();

        for _ in 0..100 {
            let counter_clone = counter.clone();
            let handle = dispatcher
                .dispatch(move || async move {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                })
                .unwrap();
            handles.push(handle);
        }

        for handle in handles {
            wait_for(handle).unwrap();
        }

        assert_eq!(counter.load(Ordering::SeqCst), 100);
        dispatcher.shutdown().unwrap();
    }

    #[test]
    fn test_task_returns_string() {
        let dispatcher = CompioDispatcher::new(2);

        let handle = dispatcher
            .dispatch(|| async { String::from("hello world") })
            .unwrap();

        let result = wait_for(handle);
        assert_eq!(result.unwrap(), "hello world");

        dispatcher.shutdown().unwrap();
    }

    #[test]
    fn test_task_returns_vec() {
        let dispatcher = CompioDispatcher::new(2);

        let handle = dispatcher
            .dispatch(|| async { vec![1, 2, 3, 4, 5] })
            .unwrap();

        let result = wait_for(handle);
        assert_eq!(result.unwrap(), vec![1, 2, 3, 4, 5]);

        dispatcher.shutdown().unwrap();
    }

    #[test]
    fn test_dispatcher_shutdown_waits_for_threads() {
        let dispatcher = CompioDispatcher::new(2);

        // Dispatch a task that takes some time
        let handle = dispatcher
            .dispatch(|| async {
                // Small delay to ensure the task is running
                std::thread::sleep(Duration::from_millis(10));
                "completed"
            })
            .unwrap();

        // Get the result before shutdown
        let result = wait_for(handle);
        assert_eq!(result.unwrap(), "completed");

        // Shutdown should complete successfully
        dispatcher.shutdown().unwrap();
    }

    #[test]
    fn test_tasks_complete_after_submitter_dropped() {
        let dispatcher = CompioDispatcher::new(2);
        let mut handles = Vec::new();

        // Submit several tasks
        for i in 0..5 {
            let handle = dispatcher.dispatch(move || async move { i }).unwrap();
            handles.push(handle);
        }

        // Shutdown will drop the submitter
        // Tasks should still complete
        std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(50));
            dispatcher.shutdown().unwrap();
        });

        // Verify all tasks complete
        for (i, handle) in handles.into_iter().enumerate() {
            let result = wait_for(handle);
            assert_eq!(result.unwrap(), i);
        }
    }

    #[test]
    fn test_empty_dispatcher_shutdown() {
        // Test that we can create and immediately shutdown a dispatcher
        let dispatcher = CompioDispatcher::new(2);
        dispatcher.shutdown().unwrap();
    }

    #[test]
    fn test_dispatcher_with_many_threads() {
        let dispatcher = CompioDispatcher::new(2);
        assert_eq!(dispatcher.threads.len(), 2);

        // Test that all threads can handle tasks
        let mut handles = Vec::new();
        for i in 0..8 {
            let handle = dispatcher.dispatch(move || async move { i }).unwrap();
            handles.push(handle);
        }

        for (i, handle) in handles.into_iter().enumerate() {
            let result = wait_for(handle);
            assert_eq!(result.unwrap(), i);
        }

        dispatcher.shutdown().unwrap();
    }

    #[test]
    fn test_task_with_delay() {
        let dispatcher = CompioDispatcher::new(2);

        let handle = dispatcher
            .dispatch(|| async {
                // Simulate async work
                futures::future::ready(()).await;
                "done"
            })
            .unwrap();

        let result = wait_for(handle);
        assert_eq!(result.unwrap(), "done");

        dispatcher.shutdown().unwrap();
    }
}
