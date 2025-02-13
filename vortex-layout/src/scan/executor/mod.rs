#[cfg(feature = "tokio")]
mod tokio;

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::channel::oneshot;
use futures::FutureExt as _;
#[cfg(feature = "tokio")]
pub use tokio::*;
use vortex_error::{vortex_err, VortexResult};

pub struct JoinHandle<T> {
    inner: oneshot::Receiver<T>,
}

impl<T> Future for JoinHandle<T> {
    type Output = VortexResult<T>;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match self.inner.poll_unpin(cx) {
            Poll::Ready(Ok(v)) => Poll::Ready(Ok(v)),
            Poll::Ready(Err(_)) => Poll::Ready(Err(vortex_err!("Task was canceled"))),
            Poll::Pending => Poll::Pending,
        }
    }
}

pub trait Spawn {
    fn spawn<F>(&self, f: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        <F as Future>::Output: Send + 'static;
}

#[derive(Default, Clone)]
pub struct InlineExecutor;

#[async_trait::async_trait]
impl Spawn for InlineExecutor {
    fn spawn<F>(&self, f: F) -> JoinHandle<F::Output>
    where
        F: Future + Send + 'static,
        <F as Future>::Output: Send + 'static,
    {
        let (tx, rx) = oneshot::channel();
        // This is very hacky and probably not a great idea, but I don't have a much better idea on how to have a sane default here.
        futures::executor::block_on(async move {
            _ = tx.send(f.await);
        });

        JoinHandle { inner: rx }
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
    fn spawn<F>(&self, f: F) -> JoinHandle<F::Output>
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
