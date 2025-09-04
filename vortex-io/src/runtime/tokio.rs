// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::future::BoxFuture;
use tokio::runtime::Handle as TokioHandle;

use crate::runtime::{AbortHandle, AbortHandleRef, Handle, Runtime};

/// A Vortex runtime that drives all work the currently scoped Tokio runtime.
pub struct TokioRuntime(TokioHandle);

impl TokioRuntime {
    pub fn new(handle: TokioHandle) -> Handle<'static> {
        Handle(Arc::new(Self(handle)))
    }

    /// Return the current Tokio runtime handle wrapped in a Vortex handle.
    pub fn handle() -> Handle<'static> {
        Handle(Arc::new(TokioRuntime(TokioHandle::current())))
    }
}

impl Runtime<'static> for TokioRuntime {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef<'static> {
        Box::new(self.0.spawn(fut).abort_handle())
    }

    fn spawn_cpu(&self, cpu: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef<'static> {
        Box::new(self.0.spawn(async move { cpu() }).abort_handle())
    }
}

impl AbortHandle<'_> for tokio::task::AbortHandle {
    fn abort(self: Box<Self>) {
        tokio::task::AbortHandle::abort(&self)
    }
}
