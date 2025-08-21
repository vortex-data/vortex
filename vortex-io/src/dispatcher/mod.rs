// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

#[cfg(feature = "compio")]
mod compio;
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

static SHARED: LazyLock<IoDispatcher> = LazyLock::new(IoDispatcher::new);

#[cfg(feature = "compio")]
use self::compio::*;
#[cfg(not(target_arch = "wasm32"))]
use self::tokio::*;
#[cfg(target_arch = "wasm32")]
use self::wasm::*;

mod sealed {
    pub trait Sealed {}

    impl Sealed for super::IoDispatcher {}

    #[cfg(feature = "compio")]
    impl Sealed for super::CompioDispatcher {}

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
    pub fn new() -> Self {
        cfg_if! {
            if #[cfg(target_arch = "wasm32")] {
                Self(Arc::new(Inner::Wasm(WasmDispatcher::new())))
            } else if #[cfg(not(feature = "compio"))] {
                Self(Arc::new(Inner::Tokio(TokioDispatcher::new(1))))
            } else {
                Self(Arc::new(Inner::Compio(CompioDispatcher::new(1))))
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

    #[cfg(feature = "compio")]
    pub fn new_compio(num_threads: usize) -> Self {
        Self(Arc::new(Inner::Compio(CompioDispatcher::new(num_threads))))
    }

    #[cfg(target_arch = "wasm32")]
    pub fn new_wasm() -> Self {
        Self(Arc::new(Inner::Wasm(WasmDispatcher)))
    }
}

impl Default for IoDispatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl IoDispatcher {
    /// Returns a handle to the current process's shared Dispatcher.
    pub fn shared() -> Self {
        SHARED.clone()
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
    #[cfg(feature = "compio")]
    Compio(CompioDispatcher),
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
            #[cfg(feature = "compio")]
            Inner::Compio(compio_dispatch) => compio_dispatch.dispatch(task),
            #[cfg(target_arch = "wasm32")]
            Inner::Wasm(wasm_dispatch) => wasm_dispatch.dispatch(task),
        }
    }

    fn shutdown(self) -> VortexResult<()> {
        if let Ok(inner) = Arc::try_unwrap(self.0) {
            match inner {
                #[cfg(not(target_arch = "wasm32"))]
                Inner::Tokio(tokio_dispatch) => tokio_dispatch.shutdown(),
                #[cfg(feature = "compio")]
                Inner::Compio(compio_dispatch) => compio_dispatch.shutdown(),
                #[cfg(target_arch = "wasm32")]
                Inner::Wasm(wasm_dispatch) => wasm_dispatch.shutdown(),
            }
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
#[cfg(not(target_arch = "wasm32"))]
#[cfg(feature = "tokio")]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    use super::{Dispatch, IoDispatcher};

    #[::tokio::test]
    async fn test_dispatcher_task_panic_handling() {
        let dispatcher = IoDispatcher::new();
        let completed = Arc::new(AtomicBool::new(false));
        let completed_clone = completed.clone();

        // Dispatch a task that will panic
        let _handle = dispatcher.dispatch(move || async move {
            panic!("Task panic");
        });

        // Also dispatch a normal task to verify dispatcher continues working
        let normal_handle = dispatcher
            .dispatch(move || async move {
                completed_clone.store(true, Ordering::SeqCst);
                42
            })
            .unwrap();

        // The panic task should propagate the error
        // Note: this depends on implementation details

        // The normal task should complete
        let result = normal_handle.await;
        assert_eq!(result.unwrap(), 42);
        assert!(completed.load(Ordering::SeqCst));

        dispatcher.shutdown().unwrap();
    }

    #[test]
    fn test_dispatcher_shutdown_empty_queue() {
        let dispatcher = IoDispatcher::new();
        // Immediate shutdown should work
        dispatcher.shutdown().unwrap();
    }

    #[::tokio::test]
    async fn test_dispatcher_many_threads() {
        let dispatcher = IoDispatcher::new();
        let mut handles = Vec::new();

        for i in 0..100 {
            let handle = dispatcher.dispatch(move || async move { i * 2 }).unwrap();
            handles.push(handle);
        }

        for (i, handle) in handles.into_iter().enumerate() {
            let result = handle.await;
            assert_eq!(result.unwrap(), i * 2);
        }

        dispatcher.shutdown().unwrap();
    }
}
