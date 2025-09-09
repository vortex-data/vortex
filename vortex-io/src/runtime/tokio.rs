// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::{Arc, LazyLock};

use futures::future::BoxFuture;
use tokio::runtime::Handle as TokioHandle;

use crate::runtime::{AbortHandle, AbortHandleRef, Handle, IoTask, Runtime};

/// A Vortex runtime that drives all work the currently scoped Tokio runtime.
pub struct TokioRuntime(TokioHandle);

impl TokioRuntime {
    pub fn with(handle: TokioHandle) -> Handle {
        Handle::new(Arc::new(Self(handle)))
    }

    /// Return the current Tokio runtime handle wrapped in a Vortex handle.
    pub fn handle() -> Handle {
        static CURRENT: LazyLock<Arc<CurrentTokioRuntime>> =
            LazyLock::new(|| Arc::new(CurrentTokioRuntime));
        Handle::new(CURRENT.clone())
    }
}

struct CurrentTokioRuntime;

impl Runtime for CurrentTokioRuntime {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        Box::new(TokioHandle::current().spawn(fut).abort_handle())
    }

    fn spawn_cpu(&self, cpu: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        Box::new(
            TokioHandle::current()
                .spawn(async move { cpu() })
                .abort_handle(),
        )
    }

    fn spawn_io(&self, task: IoTask) {
        TokioHandle::current().spawn(task.source.drive_send(task.stream));
    }
}

impl Runtime for TokioRuntime {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        Box::new(self.0.spawn(fut).abort_handle())
    }

    fn spawn_cpu(&self, cpu: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        Box::new(self.0.spawn(async move { cpu() }).abort_handle())
    }

    fn spawn_io(&self, task: IoTask) {
        self.0.spawn(task.source.drive_send(task.stream));
    }
}

impl AbortHandle for tokio::task::AbortHandle {
    fn abort(self: Box<Self>) {
        tokio::task::AbortHandle::abort(&self)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures::FutureExt;
    use futures::executor::block_on;
    use tokio::runtime::Runtime as TokioRt;

    use super::*;

    #[test]
    fn test_spawn_simple_future() {
        let tokio_rt = TokioRt::new().unwrap();
        let handle = TokioRuntime::with(tokio_rt.handle().clone());
        let result = block_on(handle.spawn(async {
            let fut = async { 77 };
            fut.await
        }));
        assert_eq!(result, 77);
    }

    #[test]
    fn test_spawn_and_abort() {
        let tokio_rt = TokioRt::new().unwrap();
        let handle = TokioRuntime::with(tokio_rt.handle().clone());

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();

        // Create a channel to ensure the future doesn't complete immediately
        let (send, recv) = tokio::sync::oneshot::channel::<()>();

        let fut = async move {
            let _ = recv.await;
            c.fetch_add(1, Ordering::SeqCst);
        };
        let task = handle.spawn(fut.boxed());
        drop(task);

        // Now we release the channel to let the future proceed if it wasn't aborted
        let _ = send.send(());
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }
}
