#[cfg(feature = "tokio")]
mod tokio;

mod threads;

use std::future::Future;

use futures::future::BoxFuture;
pub use threads::*;
#[cfg(feature = "tokio")]
pub use tokio::*;
use vortex_error::VortexResult;

pub trait Executor {
    /// Spawns a future to run on a different runtime.
    /// The runtime will make progress on the future without being directly polled, returning its output.
    fn spawn<F>(&self, f: F) -> BoxFuture<'static, VortexResult<F::Output>>
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

#[async_trait::async_trait]
impl Executor for TaskExecutor {
    fn spawn<F>(&self, f: F) -> BoxFuture<'static, VortexResult<F::Output>>
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
