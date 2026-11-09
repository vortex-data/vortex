// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::LazyLock;

#[cfg(unix)]
use custom_labels::Labelset;
#[cfg(unix)]
use custom_labels::asynchronous::Label;
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
        spawn_current(fut)
    }

    fn spawn_cpu(&self, cpu: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        spawn_current(async move { cpu() })
    }

    fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        Executor::spawn_blocking_io(&tokio::runtime::Handle::current(), task)
    }
}

fn spawn_current(fut: impl Future<Output = ()> + Send + 'static) -> AbortHandleRef {
    #[cfg(unix)]
    let fut = fut.with_current_labels();
    // Tokio borrows the scoped handle here, avoiding a shared runtime refcount update per spawn.
    Box::new(tokio::spawn(fut).abort_handle())
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
    use rstest::rstest;
    use tokio::runtime::Runtime as TokioRt;
    use tokio::sync::oneshot;
    use vortex_error::VortexResult;

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
    fn current_handle_follows_entered_runtime() -> VortexResult<()> {
        let handle = TokioRuntime::current();
        for _ in 0..2 {
            let runtime = tokio::runtime::Builder::new_current_thread().build()?;
            let id = runtime.handle().id();
            runtime.block_on(async {
                assert_eq!(
                    handle
                        .spawn(async { tokio::runtime::Handle::current().id() })
                        .await,
                    id
                );
                assert_eq!(
                    handle
                        .spawn_cpu(|| tokio::runtime::Handle::current().id())
                        .await,
                    id
                );
            });
        }
        Ok(())
    }

    #[rstest]
    #[case(false)]
    #[case(true)]
    fn test_spawn_and_abort(#[case] current: bool) {
        let tokio_rt = TokioRt::new().unwrap();
        let runtime = TokioRuntime::from(tokio_rt.handle());
        let handle = if current {
            TokioRuntime::current()
        } else {
            runtime.handle()
        };

        let counter = Arc::new(AtomicUsize::new(0));
        let task_counter = Arc::clone(&counter);

        let (started_send, started_recv) = oneshot::channel::<()>();
        let (mut release_send, release_recv) = oneshot::channel::<()>();

        let future = async move {
            started_send
                .send(())
                .expect("test retains the startup receiver");

            let _ = release_recv.await;
            task_counter.fetch_add(1, Ordering::SeqCst);
        };
        let task = {
            let _guard = tokio_rt.enter();
            handle.spawn(future.boxed())
        };

        tokio_rt
            .block_on(started_recv)
            .expect("spawned task must signal startup");

        drop(task);

        // Keeping the sender open without sending prevents normal completion.
        tokio_rt.block_on(release_send.closed());

        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }
}
