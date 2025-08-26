// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::runtime::{CpuTask, Handle, IoTask, Runtime};
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::Stream;
use futures::StreamExt;
use std::sync::Arc;
use tokio::runtime::Handle as TokioHandle;
use tokio::task::LocalSet;

/// A Vortex runtime that drives all work on a provided Tokio runtime.
#[derive(Clone)]
pub struct TokioRuntime(Arc<TokioHandle>);

impl TokioRuntime {
    pub fn new(handle: TokioHandle) -> Self {
        TokioRuntime(Arc::new(handle))
    }
}

impl Default for TokioRuntime {
    fn default() -> Self {
        Self::new(TokioHandle::current())
    }
}

impl Runtime for TokioHandle {
    fn spawn_scheduling(&self, fut: BoxFuture<'static, ()>) {
        TokioHandle::spawn(self, fut);
    }

    fn spawn_cpu(&self, f: CpuTask) {
        // We spawn CPU tasks as if they were normal async tasks on the Tokio runtime.
        TokioHandle::spawn(self, async move { f.run() });
    }

    fn spawn_io(&self, stream: BoxStream<'static, IoTask>, concurrency: usize) {
        TokioHandle::spawn(self, async move {
            let local_set = LocalSet::new();
            local_set.spawn_local(async move {
                stream
                    .map(|t: IoTask| t.run())
                    .buffer_unordered(concurrency)
                    .collect::<()>()
                    .await
            });
        });
    }
}

impl TokioRuntime {
    /// Drive the given Vortex future on the underlying Tokio runtime.
    pub fn drive<F, Fut, R>(self, f: F) -> impl Future<Output = R> + Send + 'static
    where
        F: FnOnce(Handle) -> Fut,
        Fut: Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        f(Handle(self.0))
    }

    /// Drive the given Vortex stream on the underlying Tokio runtime.
    pub fn drive_stream<F, S, R>(self, f: F) -> impl Stream<Item = R> + Send + 'static
    where
        F: FnOnce(Handle) -> S,
        S: Stream<Item = R> + Send + 'static,
        R: Send + 'static,
    {
        f(Handle(self.0))
    }
}
