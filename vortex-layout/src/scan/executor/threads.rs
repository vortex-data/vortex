use std::future::Future;
use std::num::NonZeroUsize;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

use futures::channel::oneshot;
use futures::future::BoxFuture;
use futures::{FutureExt as _, TryFutureExt as _};
use vortex_error::{vortex_err, VortexResult, VortexUnwrap};

use super::Executor;

trait Task {
    fn run(self: Box<Self>);
}

struct ExecutorTask<F, R> {
    task: F,
    result: oneshot::Sender<R>,
}

impl<F, R> Task for ExecutorTask<F, R>
where
    F: Future<Output = R> + Send,
    R: Send,
{
    fn run(self: Box<Self>) {
        let Self { task, result } = *self;
        futures::executor::block_on(async move {
            let output = task.await;
            _ = result.send(output);
        })
    }
}

/// Multithreaded task executor, runs tasks on a dedicated thread pool.
#[derive(Clone, Default)]
pub struct ThreadsExecutor {
    inner: Arc<Inner>,
}

impl ThreadsExecutor {
    pub fn new(num_threads: NonZeroUsize) -> Self {
        Self {
            inner: Arc::new(Inner::new(num_threads)),
        }
    }
}

struct Inner {
    submitter: flume::Sender<Box<dyn Task + Send>>,
    /// True as long as the runtime should be running
    is_running: Arc<AtomicBool>,
}

impl Default for Inner {
    fn default() -> Self {
        // Safety:
        // 1 isn't 0
        Self::new(unsafe { NonZeroUsize::new_unchecked(1) })
    }
}

impl Inner {
    fn new(num_threads: NonZeroUsize) -> Self {
        let (tx, rx) = flume::unbounded::<Box<dyn Task + Send>>();
        let shutdown_signal = Arc::new(AtomicBool::new(true));
        (0..num_threads.get()).for_each(|_| {
            let rx = rx.clone();
            let shutdown_signal = shutdown_signal.clone();
            std::thread::spawn(move || {
                // The channel errors if all senders are dropped, which means we probably don't care about the task anymore,
                // and we can break and let the thread end.
                while shutdown_signal.load(Ordering::Relaxed) {
                    if let Ok(task) = rx.recv() {
                        task.run()
                    } else {
                        break;
                    }
                }
            });
        });

        Self {
            submitter: tx,
            is_running: shutdown_signal,
        }
    }
}

impl Executor for ThreadsExecutor {
    fn spawn<F>(&self, f: F) -> BoxFuture<'static, VortexResult<F::Output>>
    where
        F: Future + Send + 'static,
        <F as Future>::Output: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        let task = Box::new(ExecutorTask {
            task: f,
            result: tx,
        });
        self.inner
            .submitter
            .send(task)
            .map_err(|e| vortex_err!("Failed to submit work to executor: {e}"))
            .vortex_unwrap();

        rx.map_err(|e| vortex_err!("Future canceled: {e}")).boxed()
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.is_running.store(false, Ordering::SeqCst);
    }
}
