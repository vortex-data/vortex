// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::future::BoxFuture;
use wasm_bindgen_futures::spawn_local;

use crate::runtime::{AbortHandle, AbortHandleRef, Handle, IoTask, Runtime};

/// A Vortex runtime that drives work in a WebAssembly environment.
pub struct WasmRuntime;

impl WasmRuntime {
    pub fn handle() -> Handle<'static> {
        Handle(Arc::new(WasmRuntime))
    }
}

impl Runtime<'static> for WasmRuntime {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef<'static> {
        spawn_local(fut);
        Box::new(NoOpAbortHandle)
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef<'static> {
        // TODO(ngates): we could in-theory use the abort-handle to cancel the CPU work if we
        //  are aborted before we start running.
        spawn_local(async move { task() });
        Box::new(NoOpAbortHandle)
    }

    fn spawn_io(&self, task: IoTask<'static>) {
        spawn_local(task.drive_local());
    }
}

struct NoOpAbortHandle;

impl AbortHandle<'_> for NoOpAbortHandle {
    fn abort(self: Box<Self>) {
        // No-op
    }
}
