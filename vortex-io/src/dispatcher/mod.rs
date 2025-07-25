// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "compio")]
pub mod compio;
#[cfg(feature = "tokio")]
pub mod tokio;
#[cfg(target_arch = "wasm32")]
pub mod wasm;

use std::future::Future;
use std::task::Poll;

use futures::FutureExt;
use futures::channel::oneshot;
use vortex_error::{VortexResult, vortex_err};

mod sealed {
    pub trait Sealed {}

    #[cfg(feature = "compio")]
    impl Sealed for crate::dispatcher::compio::CompioDispatcher {}

    #[cfg(feature = "tokio")]
    impl Sealed for crate::dispatcher::tokio::TokioDispatcher {}

    #[cfg(target_arch = "wasm32")]
    impl Sealed for crate::dispatcher::wasm::WasmDispatcher {}
}

/// A trait for types that may be dispatched.
pub trait Dispatch: sealed::Sealed {
    /// Dispatch a new asynchronous task.
    ///
    /// The function spawning the task must be `Send` as it will be sent to
    /// the driver thread.
    ///
    /// The returned `Future` will be executed to completion on a single thread,
    /// thus it may be `!Send`.
    fn dispatch<F, Fut, R>(&self, task: F) -> VortexResult<JoinHandle<R>>
    where
        F: (FnOnce() -> Fut) + Send + 'static,
        Fut: Future<Output = R> + 'static,
        R: Send + 'static;

    /// Gracefully shutdown the dispatcher, consuming it.
    ///
    /// Existing tasks are awaited before exiting.
    fn shutdown(self) -> VortexResult<()>;
}

pub struct JoinHandle<R>(oneshot::Receiver<R>);

impl<R> Future for JoinHandle<R> {
    type Output = VortexResult<R>;

    fn poll(
        mut self: std::pin::Pin<&mut Self>,
        cx: &mut std::task::Context<'_>,
    ) -> Poll<Self::Output> {
        match self.0.poll_unpin(cx) {
            Poll::Ready(Ok(v)) => Poll::Ready(Ok(v)),
            Poll::Ready(Err(_)) => Poll::Ready(Err(vortex_err!("Task was canceled"))),
            Poll::Pending => Poll::Pending,
        }
    }
}
