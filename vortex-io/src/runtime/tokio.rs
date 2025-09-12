// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::{Arc, LazyLock};

use futures::future::BoxFuture;
use tokio::runtime::Handle as TokioHandle;

use crate::runtime::{AbortHandle, AbortHandleRef, Handle, IoTask, Runtime};

/// A Vortex runtime that drives all work the currently scoped Tokio runtime.
#[allow(dead_code)]
pub struct TokioRuntime(Arc<TokioHandle>);

impl TokioRuntime {
    pub fn with(handle: TokioHandle) -> Handle {
        Handle::new(Arc::new(handle))
    }

    /// Return the current Tokio runtime handle wrapped in a Vortex handle.
    pub fn current() -> Handle {
        static CURRENT: LazyLock<Arc<CurrentTokioRuntime>> =
            LazyLock::new(|| Arc::new(CurrentTokioRuntime));
        Handle::new(CURRENT.clone())
    }
}

/// A runtime implementation that grabs the current Tokio runtime handle on each call.
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

impl Runtime for TokioHandle {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        Box::new(TokioHandle::spawn(self, fut).abort_handle())
    }

    fn spawn_cpu(&self, cpu: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        Box::new(TokioHandle::spawn(self, async move { cpu() }).abort_handle())
    }

    fn spawn_io(&self, task: IoTask) {
        TokioHandle::spawn(self, task.source.drive_send(task.stream));
    }
}

impl AbortHandle for tokio::task::AbortHandle {
    fn abort(self: Box<Self>) {
        tokio::task::AbortHandle::abort(&self)
    }
}

// We depend on Tokio's rt-multi-thread feature for block-in-place
#[cfg(feature = "tokio")]
impl crate::runtime::BlockingRuntime for TokioRuntime {
    type BlockingIterator<'a, R: 'a> = TokioBlockingIterator<'a, R>;

    fn handle(&self) -> Handle {
        Handle::new(self.0.clone())
    }

    fn block_on<Fut, R>(&self, fut: Fut) -> R
    where
        Fut: Future<Output = R>,
    {
        // Assert that we're not currently inside the Tokio context.
        if TokioHandle::try_current().is_ok() {
            vortex_error::vortex_panic!("block_on cannot be called from within a Tokio runtime");
        }
        let handle = self.0.clone();
        tokio::task::block_in_place(move || handle.block_on(fut))
    }

    fn block_on_stream<'a, S, R>(&self, stream: S) -> Self::BlockingIterator<'a, R>
    where
        S: futures::Stream<Item = R> + Send + Unpin + 'a,
        R: Send + 'a,
    {
        // Assert that we're not currently inside the Tokio context.
        if TokioHandle::try_current().is_ok() {
            vortex_error::vortex_panic!(
                "block_on_stream cannot be called from within a Tokio runtime"
            );
        }
        let handle = self.0.clone();
        let stream = Box::pin(stream);
        TokioBlockingIterator { handle, stream }
    }
}

#[cfg(feature = "tokio")]
pub struct TokioBlockingIterator<'a, T> {
    handle: Arc<TokioHandle>,
    stream: futures::stream::BoxStream<'a, T>,
}

#[cfg(feature = "tokio")]
impl<T> Iterator for TokioBlockingIterator<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        use futures::StreamExt;

        tokio::task::block_in_place(|| self.handle.block_on(self.stream.next()))
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
