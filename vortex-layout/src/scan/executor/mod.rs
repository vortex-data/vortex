#[cfg(feature = "tokio")]
mod tokio;

mod threads;

use std::future::Future;

use futures::future::BoxFuture;
pub use threads::*;
#[cfg(feature = "tokio")]
pub use tokio::*;
use vortex_error::VortexResult;

pub trait Spawn {
    // Spawns a future to run on a different runtime. The returning future should be polled to ensure its running.
    fn spawn<F>(&self, f: F) -> VortexResult<BoxFuture<'static, VortexResult<F::Output>>>
    where
        F: Future + Send + 'static,
        <F as Future>::Output: Send + 'static;
}

/// Generic wrapper around different async runtimes. Can be used to spawn futures to run in the background, concurrently with other tasks.
#[derive(Clone)]
pub enum Executor {
    Threads(ThreadsExecutor),
    #[cfg(feature = "tokio")]
    Tokio(TokioExecutor),
}

#[async_trait::async_trait]
impl Spawn for Executor {
    fn spawn<F>(&self, f: F) -> VortexResult<BoxFuture<'static, VortexResult<F::Output>>>
    where
        F: Future + Send + 'static,
        <F as Future>::Output: Send + 'static,
    {
        match self {
            Executor::Threads(threads_executor) => threads_executor.spawn(f),
            #[cfg(feature = "tokio")]
            Executor::Tokio(tokio_executor) => tokio_executor.spawn(f),
        }
    }
}
