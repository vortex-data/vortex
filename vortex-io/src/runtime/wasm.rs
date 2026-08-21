// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use futures::future::BoxFuture;
use futures::future::abortable;
use wasm_bindgen_futures::spawn_local;

use crate::runtime::AbortHandle;
use crate::runtime::AbortHandleRef;
use crate::runtime::Executor;
use crate::runtime::Handle;

/// A Vortex runtime that drives work on the JavaScript event loop.
///
/// This requires the `wasm-bindgen` feature. In a WebAssembly environment with no JavaScript
/// event loop, drive a [`crate::runtime::single::SingleThreadRuntime`] instead.
pub struct WasmRuntime;

impl WasmRuntime {
    pub fn handle() -> Handle {
        static RUNTIME: LazyLock<Arc<dyn Executor>> = LazyLock::new(|| Arc::new(WasmRuntime));

        Handle::new(Arc::downgrade(&RUNTIME))
    }
}

impl Executor for WasmRuntime {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        // `spawn_local` is fire-and-forget, so make the future abortable to keep the
        // cancel-on-drop semantics of the other runtimes.
        let (fut, handle) = abortable(fut);
        spawn_local(async move {
            let _ = fut.await;
        });
        Box::new(FuturesAbortHandle(handle))
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        spawn_local(async move { task() });
        Box::new(NoOpAbortHandle)
    }

    fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        spawn_local(async move { task() });
        Box::new(NoOpAbortHandle)
    }
}

/// An abort handle for a future spawned onto the JavaScript event loop.
///
/// [`futures::future::AbortHandle`] already matches this crate's semantics: dropping it detaches
/// the future, while aborting wakes it so the event loop drops it on the next poll.
struct FuturesAbortHandle(futures::future::AbortHandle);

impl AbortHandle for FuturesAbortHandle {
    fn abort(self: Box<Self>) {
        self.0.abort();
    }
}

struct NoOpAbortHandle;

impl AbortHandle for NoOpAbortHandle {
    fn abort(self: Box<Self>) {
        // No-op
    }
}
