// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use futures::future::BoxFuture;
use wasm_bindgen_futures::spawn_local;

use crate::runtime::AbortHandle;
use crate::runtime::AbortHandleRef;
use crate::runtime::Executor;
use crate::runtime::Handle;

/// A Vortex runtime that drives work in a WebAssembly environment.
pub struct WasmRuntime;

impl WasmRuntime {
    pub fn handle() -> Handle {
        static RUNTIME: LazyLock<Arc<dyn Executor>> = LazyLock::new(|| Arc::new(WasmRuntime));

        Handle::new(Arc::downgrade(&RUNTIME))
    }
}

impl Executor for WasmRuntime {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        spawn_local(fut);
        Box::new(NoOpAbortHandle)
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        // TODO(ngates): we could in-theory use the abort-handle to cancel the CPU work if we
        //  are aborted before we start running.
        spawn_local(async move { task() });
        Box::new(NoOpAbortHandle)
    }

    fn spawn_blocking(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        spawn_local(async move { task() });
        Box::new(NoOpAbortHandle)
    }
}

struct NoOpAbortHandle;

impl AbortHandle for NoOpAbortHandle {
    fn abort(self: Box<Self>) {
        // No-op
    }
}
