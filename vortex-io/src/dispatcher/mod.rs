#[cfg(feature = "compio")]
mod compio;
#[cfg(feature = "tokio")]
mod tokio;
#[cfg(target_arch = "wasm32")]
mod wasm;

use std::future::Future;
use std::task::Poll;

use futures::channel::oneshot;
use futures::FutureExt;
#[cfg(not(any(feature = "compio", feature = "tokio", target_arch = "wasm32")))]
use vortex_error::vortex_panic;
use vortex_error::{vortex_err, VortexResult};

#[cfg(feature = "compio")]
use self::compio::*;
#[cfg(feature = "tokio")]
use self::tokio::*;
#[cfg(target_arch = "wasm32")]
use self::wasm::*;

mod sealed {
    pub trait Sealed {}

    impl Sealed for super::IoDispatcher {}

    #[cfg(feature = "compio")]
    impl Sealed for super::CompioDispatcher {}

    #[cfg(feature = "tokio")]
    impl Sealed for super::TokioDispatcher {}

    #[cfg(target_arch = "wasm32")]
    impl Sealed for super::WasmDispatcher {}
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

/// <div class="warning">IoDispatcher is unstable and may change in the future.</div>
///
/// A cross-thread, cross-runtime dispatcher of async IO workloads.
///
/// `IoDispatcher`s are handles to an async runtime that can handle work submissions and
/// multiplexes them across a set of worker threads. Unlike an async runtime, which is free
/// to balance tasks as they see fit, the purpose of the Dispatcher is to enable the spawning
/// of asynchronous, `!Send` tasks across potentially many worker threads, and allowing work
/// submission from any other runtime.
///
#[derive(Debug)]
pub struct IoDispatcher(Inner);

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

#[derive(Debug)]
enum Inner {
    #[cfg(feature = "tokio")]
    Tokio(TokioDispatcher),
    #[cfg(feature = "compio")]
    Compio(CompioDispatcher),
    #[cfg(target_arch = "wasm32")]
    Wasm(WasmDispatcher),
}

impl Default for IoDispatcher {
    #[cfg(target_arch = "wasm32")]
    fn default() -> Self {
        return Self(Inner::Wasm(WasmDispatcher::new()));
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn default() -> Self {
        #[cfg(feature = "tokio")]
        return Self(Inner::Tokio(TokioDispatcher::new(1)));
        #[cfg(all(feature = "compio", not(feature = "tokio")))]
        return Self(Inner::Compio(CompioDispatcher::new(1)));
        #[cfg(not(any(feature = "compio", feature = "tokio")))]
        vortex_panic!("must enable one of compio or tokio to use IoDispatcher");
    }
}

impl Dispatch for IoDispatcher {
    #[allow(unused_variables)] // If no features are enabled `task` ends up being unused
    fn dispatch<F, Fut, R>(&self, task: F) -> VortexResult<JoinHandle<R>>
    where
        F: (FnOnce() -> Fut) + Send + 'static,
        Fut: Future<Output = R> + 'static,
        R: Send + 'static,
    {
        match self.0 {
            #[cfg(feature = "tokio")]
            Inner::Tokio(ref tokio_dispatch) => tokio_dispatch.dispatch(task),
            #[cfg(feature = "compio")]
            Inner::Compio(ref compio_dispatch) => compio_dispatch.dispatch(task),
            #[cfg(target_arch = "wasm32")]
            Inner::Wasm(ref wasm_dispatch) => wasm_dispatch.dispatch(task),
        }
    }

    fn shutdown(self) -> VortexResult<()> {
        match self.0 {
            #[cfg(feature = "tokio")]
            Inner::Tokio(tokio_dispatch) => tokio_dispatch.shutdown(),
            #[cfg(feature = "compio")]
            Inner::Compio(compio_dispatch) => compio_dispatch.shutdown(),
            #[cfg(target_arch = "wasm32")]
            Inner::Wasm(wasm_dispatch) => wasm_dispatch.shutdown(),
        }
    }
}

impl IoDispatcher {
    /// Create a new IO dispatcher that uses a set of Tokio `current_thread` runtimes to
    /// execute both `Send` and `!Send` futures.
    ///
    /// A handle to the dispatcher can be passed freely among threads, allowing multiple parties to
    /// perform dispatching across different threads.
    #[cfg(feature = "tokio")]
    pub fn new_tokio(num_thread: usize) -> Self {
        Self(Inner::Tokio(TokioDispatcher::new(num_thread)))
    }

    #[cfg(feature = "compio")]
    pub fn new_compio(num_threads: usize) -> Self {
        Self(Inner::Compio(CompioDispatcher::new(num_threads)))
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new_wasm() -> Self {
        Self(Inner::Wasm(WasmDispatcher))
    }
}
