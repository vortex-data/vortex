#[cfg(feature = "tokio")]
mod tokio;

mod threads;

use std::fmt::{Debug, Formatter};
use std::future::Future;

use futures::future::BoxFuture;
pub use threads::*;
#[cfg(feature = "tokio")]
pub use tokio::*;

pub trait Executor {
    /// Spawns a future to run on a different runtime.
    /// The runtime will make progress on the future without being directly polled, returning its output.
    fn spawn<F>(&self, f: F) -> BoxFuture<'static, F::Output>
    where
        F: Future + Send + 'static,
        <F as Future>::Output: Send + 'static;
}

/// Generic wrapper around different async runtimes. Used to spawn async tasks to run in the background, concurrently with other tasks.
#[derive(Clone)]
pub enum TaskExecutor {
    Threads(ThreadsExecutor),
    #[cfg(feature = "tokio")]
    Tokio(TokioExecutor),
}

impl Debug for TaskExecutor {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TaskExecutor")
            .field(
                "variant",
                &match self {
                    TaskExecutor::Threads(_) => "Threads",
                    #[cfg(feature = "tokio")]
                    TaskExecutor::Tokio(_) => "Tokio",
                },
            )
            .finish()
    }
}

#[async_trait::async_trait]
impl Executor for TaskExecutor {
    fn spawn<F>(&self, f: F) -> BoxFuture<'static, F::Output>
    where
        F: Future + Send + 'static,
        <F as Future>::Output: Send + 'static,
    {
        match self {
            TaskExecutor::Threads(threads_executor) => threads_executor.spawn(f),
            #[cfg(feature = "tokio")]
            TaskExecutor::Tokio(tokio_executor) => tokio_executor.spawn(f),
        }
    }
}
