// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! The dispatcher module provides an abstract way of launching background I/O-bound tasks. These
//! tasks can be `!Send` and are executed on a dedicated thread pool.
//!
//! It is important to note that the tasks must still be runtime agnostic (i.e., they should not
//! depend on a specific runtime's APIs). The [`crate::ReadAt`] trait provides a way of creating
//! runtime agnostic I/O futures.

#[cfg(not(target_arch = "wasm32"))]
mod tokio;
#[cfg(target_arch = "wasm32")]
mod wasm;

use std::future::Future;
use std::sync::{Arc, LazyLock};
use std::task::Poll;

use cfg_if::cfg_if;
use futures::FutureExt;
use futures::channel::oneshot;
use vortex_error::{VortexResult, vortex_err};

/// The global shared I/O dispatcher.
// TODO(ngates): we may want just an instance of this inside VortexSession.
static DISPATCHER: LazyLock<IoDispatcher> = LazyLock::new(IoDispatcher::new);

#[cfg(not(target_arch = "wasm32"))]
use self::tokio::*;
#[cfg(target_arch = "wasm32")]
use self::wasm::*;

mod sealed {
    pub trait Sealed {}

    impl Sealed for super::IoDispatcher {}

    #[cfg(not(target_arch = "wasm32"))]
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
#[derive(Clone, Debug)]
pub struct IoDispatcher(Arc<Inner>);

impl IoDispatcher {
    fn new() -> Self {
        cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                Self(Arc::new(Inner::Wasm(WasmDispatcher::new())))
            } else {
                Self(Arc::new(Inner::Tokio(TokioDispatcher::new(1))))
            }
        }
    }

    /// Create a new IO dispatcher that uses a set of Tokio `current_thread` runtimes to
    /// execute both `Send` and `!Send` futures.
    ///
    /// A handle to the dispatcher can be passed freely among threads, allowing multiple parties to
    /// perform dispatching across different threads.
    #[cfg(not(target_arch = "wasm32"))]
    pub fn new_tokio(num_thread: usize) -> Self {
        Self(Arc::new(Inner::Tokio(TokioDispatcher::new(num_thread))))
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new_wasm() -> Self {
        Self(Arc::new(Inner::Wasm(WasmDispatcher)))
    }
}

impl IoDispatcher {
    /// Returns a handle to the current process's shared Dispatcher.
    pub fn shared() -> Self {
        DISPATCHER.clone()
    }
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

#[derive(Debug)]
enum Inner {
    #[cfg(not(target_arch = "wasm32"))]
    Tokio(TokioDispatcher),
    #[cfg(target_arch = "wasm32")]
    Wasm(WasmDispatcher),
}

impl Dispatch for IoDispatcher {
    #[allow(unused_variables)] // If no features are enabled `task` ends up being unused
    fn dispatch<F, Fut, R>(&self, task: F) -> VortexResult<JoinHandle<R>>
    where
        F: (FnOnce() -> Fut) + Send + 'static,
        Fut: Future<Output = R> + 'static,
        R: Send + 'static,
    {
        match self.0.as_ref() {
            #[cfg(not(target_arch = "wasm32"))]
            Inner::Tokio(tokio_dispatch) => tokio_dispatch.dispatch(task),
            #[cfg(target_arch = "wasm32")]
            Inner::Wasm(wasm_dispatch) => wasm_dispatch.dispatch(task),
        }
    }

    fn shutdown(self) -> VortexResult<()> {
        if let Ok(inner) = Arc::try_unwrap(self.0) {
            match inner {
                #[cfg(not(target_arch = "wasm32"))]
                Inner::Tokio(tokio_dispatch) => tokio_dispatch.shutdown(),
                #[cfg(target_arch = "wasm32")]
                Inner::Wasm(wasm_dispatch) => wasm_dispatch.shutdown(),
            }
        } else {
            Ok(())
        }
    }
}
