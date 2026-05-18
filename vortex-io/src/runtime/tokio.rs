// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

use futures::future::BoxFuture;

use crate::runtime::AbortHandle;
use crate::runtime::AbortHandleRef;
use crate::runtime::BlockingRuntime;
use crate::runtime::Executor;
use crate::runtime::Handle;

/// A Vortex runtime that drives all work the enclosed Tokio runtime handle.
pub struct TokioRuntime(Arc<tokio::runtime::Handle>);

impl TokioRuntime {
    pub fn new(handle: tokio::runtime::Handle) -> Self {
        Self(Arc::new(handle))
    }

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
        #[cfg(unix)]
        {
            use custom_labels::asynchronous::Label;

            let fut = fut.with_current_labels();
            Box::new(tokio::runtime::Handle::spawn(self, fut).abort_handle())
        }
        #[cfg(not(unix))]
        {
            Box::new(tokio::runtime::Handle::spawn(self, fut).abort_handle())
        }
    }

    fn spawn_cpu(&self, cpu: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        #[cfg(unix)]
        {
            use custom_labels::asynchronous::Label;

            Box::new(
                tokio::runtime::Handle::spawn(self, async move { cpu() }.with_current_labels())
                    .abort_handle(),
            )
        }
        #[cfg(not(unix))]
        {
            Box::new(tokio::runtime::Handle::spawn(self, async move { cpu() }).abort_handle())
        }
    }

    fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        #[cfg(unix)]
        {
            use custom_labels::Labelset;

            let mut set = Labelset::clone_from_current();
            Box::new(
                tokio::runtime::Handle::spawn_blocking(self, move || set.enter(task))
                    .abort_handle(),
            )
        }
        #[cfg(not(unix))]
        {
            Box::new(tokio::runtime::Handle::spawn_blocking(self, task).abort_handle())
        }
    }
}

/// A runtime implementation that grabs the current Tokio runtime handle on each call.
struct CurrentTokioRuntime;

impl Executor for CurrentTokioRuntime {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        Executor::spawn(&tokio::runtime::Handle::current(), fut)
    }

    fn spawn_cpu(&self, cpu: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        Executor::spawn_cpu(&tokio::runtime::Handle::current(), cpu)
    }

    fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        Executor::spawn_blocking_io(&tokio::runtime::Handle::current(), task)
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
        let executor: Arc<dyn Executor> = Arc::clone(&self.0) as Arc<dyn Executor>;
        Handle::new(Arc::downgrade(&executor))
    }

    fn block_on<Fut, R>(&self, fut: Fut) -> R
    where
        Fut: Future<Output = R>,
    {
        // Assert that we're not currently inside the Tokio context.
        if tokio::runtime::Handle::try_current().is_ok() {
            vortex_error::vortex_panic!("block_on cannot be called from within a Tokio runtime");
        }
        let handle = Arc::clone(&self.0);
        tokio::task::block_in_place(move || handle.block_on(fut))
    }

    fn block_on_stream<'a, S, R>(&self, stream: S) -> Self::BlockingIterator<'a, R>
    where
        S: futures::Stream<Item = R> + Send + 'a,
        R: Send + 'a,
    {
        // Assert that we're not currently inside the Tokio context.
        if tokio::runtime::Handle::try_current().is_ok() {
            vortex_error::vortex_panic!(
                "block_on_stream cannot be called from within a Tokio runtime"
            );
        }
        let handle = Arc::clone(&self.0);
        let stream = Box::pin(stream);
        TokioBlockingIterator { handle, stream }
    }
}

pub struct TokioBlockingIterator<'a, T> {
    handle: Arc<tokio::runtime::Handle>,
    stream: futures::stream::BoxStream<'a, T>,
}

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
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::FutureExt;
    use tokio::runtime::Runtime as TokioRt;
    use tokio::sync::oneshot;

    use super::*;

    #[test]
    fn test_spawn_simple_future() {
        let tokio_rt = TokioRt::new().unwrap();
        let runtime = TokioRuntime::from(tokio_rt.handle());
        let h = runtime.handle();
        let result = runtime.block_on({
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
        let c = Arc::clone(&counter);

        // Create a channel to ensure the future doesn't complete immediately
        let (send, recv) = oneshot::channel::<()>();

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
