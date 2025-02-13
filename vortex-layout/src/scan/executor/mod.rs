#[cfg(feature = "tokio")]
mod tokio;

use std::future::Future;

use futures::future::BoxFuture;
use futures::FutureExt as _;
#[cfg(feature = "tokio")]
pub use tokio::*;
use vortex_error::VortexResult;

pub trait Spawn {
    fn spawn<F>(&self, f: F) -> BoxFuture<'static, VortexResult<F::Output>>
    where
        F: Future + Send + 'static,
        <F as Future>::Output: Send + 'static;
}

#[derive(Default, Clone)]
pub struct InlineExecutor;

#[async_trait::async_trait]
impl Spawn for InlineExecutor {
    fn spawn<F>(&self, f: F) -> BoxFuture<'static, VortexResult<F::Output>>
    where
        F: Future + Send + 'static,
        <F as Future>::Output: Send + 'static,
    {
        async move { Ok(f.await) }.boxed()
    }
}

#[derive(Clone)]
pub enum Executor {
    Inline(InlineExecutor),
    #[cfg(feature = "tokio")]
    Tokio(TokioExecutor),
}

#[async_trait::async_trait]
impl Spawn for Executor {
    fn spawn<F>(&self, f: F) -> BoxFuture<'static, VortexResult<F::Output>>
    where
        F: Future + Send + 'static,
        <F as Future>::Output: Send + 'static,
    {
        match self {
            Executor::Inline(inline_executor) => inline_executor.spawn(f),
            #[cfg(feature = "tokio")]
            Executor::Tokio(tokio_executor) => tokio_executor.spawn(f),
        }
    }
}
