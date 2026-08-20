// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use futures::future::BoxFuture;
#[cfg(feature = "wasm-bindgen")]
use wasm_bindgen_futures::spawn_local;

use crate::runtime::AbortHandle;
use crate::runtime::AbortHandleRef;
use crate::runtime::Executor;
use crate::runtime::Handle;
#[cfg(not(feature = "wasm-bindgen"))]
use crate::runtime::inline::block_on;

/// A Vortex runtime that drives work in a browser WebAssembly environment.
///
/// With the `wasm-bindgen` feature, tasks are scheduled on the JavaScript event loop. Without it,
/// tasks run synchronously and must not depend on yielding control to an external event loop.
pub struct WasmRuntime;

impl WasmRuntime {
    pub fn handle() -> Handle {
        static RUNTIME: LazyLock<Arc<dyn Executor>> = LazyLock::new(|| Arc::new(WasmRuntime));

        Handle::new(Arc::downgrade(&RUNTIME))
    }
}

impl Executor for WasmRuntime {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        #[cfg(feature = "wasm-bindgen")]
        spawn_local(fut);
        #[cfg(not(feature = "wasm-bindgen"))]
        block_on(fut);
        Box::new(NoOpAbortHandle)
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        // TODO(ngates): we could in-theory use the abort-handle to cancel the CPU work if we
        //  are aborted before we start running.
        #[cfg(feature = "wasm-bindgen")]
        spawn_local(async move { task() });
        #[cfg(not(feature = "wasm-bindgen"))]
        task();
        Box::new(NoOpAbortHandle)
    }

    fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        #[cfg(feature = "wasm-bindgen")]
        spawn_local(async move { task() });
        #[cfg(not(feature = "wasm-bindgen"))]
        task();
        Box::new(NoOpAbortHandle)
    }
}

struct NoOpAbortHandle;

impl AbortHandle for NoOpAbortHandle {
    fn abort(self: Box<Self>) {
        // No-op
    }
}
