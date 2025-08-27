// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{CpuTask, Handle, IoTask, Runtime};
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::StreamExt;
use std::sync::Arc;
use tokio::runtime::Handle as TokioHandle;

/// A Vortex runtime that drives all work the currently scoped Tokio runtime.
pub struct TokioRuntime(TokioHandle);

impl TokioRuntime {
    /// Return the current Tokio runtime handle wrapped in a Vortex handle.
    pub fn handle() -> Handle<'static> {
        Handle(Arc::new(TokioRuntime(TokioHandle::current())))
    }
}

impl Runtime<'static> for TokioRuntime {
    fn spawn_scheduling(&self, fut: BoxFuture<'static, ()>) {
        self.0.spawn(fut);
    }

    fn spawn_cpu(&self, f: CpuTask) {
        // We spawn CPU tasks as if they were normal async tasks on the Tokio runtime.
        self.0.spawn(async move { f.run() });
    }

    fn spawn_io(&self, stream: BoxStream<'static, IoTask>, concurrency: usize) {
        self.0.spawn(async move {
            stream
                .map(|t: IoTask| t.run_send())
                .buffer_unordered(concurrency)
                .collect::<()>()
                .await
        });
    }
}
