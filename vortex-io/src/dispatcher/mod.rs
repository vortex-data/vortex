#[cfg(feature = "compio")]
mod compio;
#[cfg(feature = "tokio")]
mod tokio;
use std::future::Future;

use futures::channel::oneshot;
use vortex_error::VortexResult;

#[cfg(feature = "compio")]
use self::compio::*;
#[cfg(feature = "tokio")]
use self::tokio::*;

mod sealed {
    pub trait Sealed {}

    impl Sealed for super::IoDispatcher {}

    #[cfg(feature = "compio")]
    impl Sealed for super::CompioDispatcher {}

    #[cfg(feature = "tokio")]
    impl Sealed for super::TokioDispatcher {}
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
    fn dispatch<F, Fut, R>(&self, task: F) -> VortexResult<oneshot::Receiver<R>>
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

#[derive(Debug)]
enum Inner {
    #[cfg(feature = "tokio")]
    Tokio(TokioDispatcher),
    #[cfg(feature = "compio")]
    Compio(CompioDispatcher),
}

impl Dispatch for IoDispatcher {
    #[allow(unused_variables)]
    fn dispatch<F, Fut, R>(&self, task: F) -> VortexResult<oneshot::Receiver<R>>
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
        }
    }

    fn shutdown(self) -> VortexResult<()> {
        match self.0 {
            #[cfg(feature = "tokio")]
            Inner::Tokio(tokio_dispatch) => tokio_dispatch.shutdown(),
            #[cfg(feature = "compio")]
            Inner::Compio(compio_dispatch) => compio_dispatch.shutdown(),
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
}
