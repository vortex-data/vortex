// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::{Arc, LazyLock};

use futures::future::BoxFuture;
use tracing::Instrument;

use crate::runtime::{AbortHandle, AbortHandleRef, BlockingRuntime, Executor, Handle, IoTask};

/// A Vortex runtime that drives all work the enclosed Tokio runtime handle.
pub struct TokioRuntime(Arc<tokio::runtime::Handle>);

impl TokioRuntime {
    /// Create a new [`Handle`] that always uses the currently scoped Tokio runtime at the time
    /// each operation is invoked.
    pub fn current() -> Handle {
        static CURRENT: LazyLock<Arc<dyn Executor>> =
            LazyLock::new(|| Arc::new(CurrentTokioRuntime));
        Handle::new(Arc::downgrade(&CURRENT))
    }
}

impl From<&tokio::runtime::Handle> for TokioRuntime {
    fn from(value: &tokio::runtime::Handle) -> Self {
        Self::from(value.clone())
    }
}

impl From<tokio::runtime::Handle> for TokioRuntime {
    fn from(value: tokio::runtime::Handle) -> Self {
        TokioRuntime(Arc::new(value))
    }
}

impl Executor for tokio::runtime::Handle {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        Box::new(tokio::runtime::Handle::spawn(self, fut).abort_handle())
    }

    fn spawn_cpu(&self, cpu: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        Box::new(tokio::runtime::Handle::spawn(self, async move { cpu() }).abort_handle())
    }

    fn spawn_blocking(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        Box::new(tokio::runtime::Handle::spawn_blocking(self, task).abort_handle())
    }

    fn spawn_io(&self, task: IoTask) {
        tokio::runtime::Handle::spawn(self, task.source.drive_send(task.stream).in_current_span());
    }
}

/// A runtime implementation that grabs the current Tokio runtime handle on each call.
struct CurrentTokioRuntime;

impl Executor for CurrentTokioRuntime {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        Box::new(tokio::runtime::Handle::current().spawn(fut).abort_handle())
    }

    fn spawn_cpu(&self, cpu: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        Box::new(
            tokio::runtime::Handle::current()
                .spawn(async move { cpu() })
                .abort_handle(),
        )
    }

    fn spawn_blocking(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        Box::new(
            tokio::runtime::Handle::current()
                .spawn_blocking(task)
                .abort_handle(),
        )
    }

    fn spawn_io(&self, task: IoTask) {
        tokio::runtime::Handle::current()
            .spawn(task.source.drive_send(task.stream).in_current_span());
    }
}

impl AbortHandle for tokio::task::AbortHandle {
    fn abort(self: Box<Self>) {
        tokio::task::AbortHandle::abort(&self)
    }
}

// We depend on Tokio's rt-multi-thread feature for block-in-place
impl BlockingRuntime for TokioRuntime {
    type BlockingIterator<'a, R: 'a> = TokioBlockingIterator<'a, R>;

    fn handle(&self) -> Handle {
        let executor: Arc<dyn Executor> = self.0.clone();
        Handle::new(Arc::downgrade(&executor))
    }

    fn block_on<F, Fut, R>(&self, f: F) -> R
    where
        F: FnOnce(Handle) -> Fut,
        Fut: Future<Output = R>,
    {
        // Assert that we're not currently inside the Tokio context.
        if tokio::runtime::Handle::try_current().is_ok() {
            vortex_error::vortex_panic!("block_on cannot be called from within a Tokio runtime");
        }
        let handle = self.0.clone();
        let fut = f(self.handle());
        tokio::task::block_in_place(move || handle.block_on(fut))
    }

    fn block_on_stream<'a, F, S, R>(&self, f: F) -> Self::BlockingIterator<'a, R>
    where
        F: FnOnce(Handle) -> S,
        S: futures::Stream<Item = R> + Send + 'a,
        R: Send + 'a,
    {
        // Assert that we're not currently inside the Tokio context.
        if tokio::runtime::Handle::try_current().is_ok() {
            vortex_error::vortex_panic!(
                "block_on_stream cannot be called from within a Tokio runtime"
            );
        }
        let handle = self.0.clone();
        let stream = Box::pin(f(self.handle()));
        TokioBlockingIterator { handle, stream }
    }
}

#[cfg(feature = "tokio")]
pub struct TokioBlockingIterator<'a, T> {
    handle: Arc<tokio::runtime::Handle>,
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
    use tokio::runtime::Runtime as TokioRt;

    use super::*;

    #[test]
    fn test_spawn_simple_future() {
        let tokio_rt = TokioRt::new().unwrap();
        let runtime = TokioRuntime::from(tokio_rt.handle());
        let result = runtime.block_on(|h| {
            h.spawn(async {
                let fut = async { 77 };
                fut.await
            })
        });
        assert_eq!(result, 77);
    }

    #[test]
    fn test_spawn_and_abort() {
        let tokio_rt = TokioRt::new().unwrap();
        let runtime = TokioRuntime::from(tokio_rt.handle());

        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();

        // Create a channel to ensure the future doesn't complete immediately
        let (send, recv) = tokio::sync::oneshot::channel::<()>();

        let fut = async move {
            let _ = recv.await;
            c.fetch_add(1, Ordering::SeqCst);
        };
        let task = runtime.handle().spawn(fut.boxed());
        drop(task);

        // Now we release the channel to let the future proceed if it wasn't aborted
        let _ = send.send(());
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }
}
