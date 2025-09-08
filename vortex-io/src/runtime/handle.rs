// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::marker::PhantomData;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, ready};

use futures::{FutureExt, StreamExt};
use vortex_error::{VortexResult, vortex_panic};

use crate::file::{FileRead, IntoIoSource, IoRequestStream};
use crate::kanal_ext::KanalExt;
use crate::runtime::{AbortHandle, IoTask, Runtime};

/// A handle represents scoped access to spawn work onto an active Vortex runtime.
///
/// This model enforces structured concurrency where all spawned tasks must complete before the
/// end of the scope. Only for handles with a `'static` scope may tasks be detached to run in the
/// background.
///
/// Users should obtain an initial handle from one of the runtime constructors and use it to spawn
/// new async tasks or CPU-heavy work.
#[derive(Clone)]
pub struct Handle<'scope, 'rt> {
    runtime: Arc<dyn Runtime<'rt> + 'rt>,
    _scope: PhantomData<&'scope ()>,
}

impl<'rt> Handle<'rt, 'rt> {
    // Create a new top-level handle where the scope == runtime lifetime.
    pub(crate) fn new(runtime: Arc<dyn Runtime<'rt> + 'rt>) -> Self {
        Self {
            runtime,
            _scope: PhantomData,
        }
    }
}

impl<'scope, 'rt> Handle<'scope, 'rt> where 'rt: 'scope
{
    /// Enter a new nested scope. All tasks spawned with the provided handle will have to complete
    /// or be dropped before this function returns.
    pub async fn scope<'child_scope, F, Fut, R>(&self, f: F) -> R
    where
        F: (FnOnce(Handle<'child_scope, 'rt>) -> Fut) + Send + 'child_scope,
        Fut: Future<Output=R> + Send + 'child_scope,
        'scope: 'child_scope,  // Parent scope outlives child
    {
        // Create handle for the child scope
        let child_handle: Handle<'child_scope, 'rt> = Handle {
            runtime: self.runtime.clone(),
            _scope: PhantomData,
        };

        // Execute the closure with the child handle, and await the future.
        let result = f(child_handle).await;

        // When we return, all tasks spawned in 'child_scope must be complete
        // This is enforced by the Task<'child_scope, _> types.
        for task in self.tasks {
            task.await_termination().await
        }
        // FIXME(ngates): we should track spawned tasks and ensure they are complete here.

        result
    }

    /// Spawns work onto the runtime within the current scope.
    ///
    /// These futures are expected to not perform expensive CPU work and instead simply schedule
    /// either CPU tasks or I/O tasks. See [`Handle::spawn_cpu`] for spawning CPU-bound work.
    ///
    /// See [`Task`] for details on cancelling or detaching the spawned task.
    pub fn spawn<Fut, R>(&self, f: Fut) -> Task<'scope, R>
    where
        Fut: Future<Output=R> + Send + 'scope,
        R: Send + 'scope,
    {
        let (send, recv) = oneshot::channel();

        // Create a future with the narrowed handle
        let fut = async move {
            // Task::detach allows the receiver to be dropped, so we ignore send errors.
            let _ = send.send(f.await);
        };

        // Extend lifetime to 'rt for the runtime
        // SAFETY: Task<'scope, R> ensures it's awaited within 'scope
        let extended = unsafe {
            std::mem::transmute::<
                Pin<Box<dyn Future<Output=()> + Send + 'scope>>,
                Pin<Box<dyn Future<Output=()> + Send + 'rt>>,
            >(Box::pin(fut))
        };

        let abort_handle = self.runtime.spawn(extended);

        // Shorten lifetime back to 'scope for the Task
        // SAFETY: Task<'scope, R> ensures it's awaited within 'scope
        let shortened = unsafe {
            std::mem::transmute::<
                Box<dyn AbortHandle + 'rt>,
                Box<dyn AbortHandle + 'scope>,
            >(abort_handle)
        };

        Task {
            recv,
            abort_handle: Some(shortened),
        }
    }

    /// Spawn a CPU-bound task for execution on the runtime.
    ///
    /// Note that many runtimes will interleave this work on the same async runtime. See the
    /// documentation for each runtime implementation for details.
    ///
    /// See [`Task`] for details on cancelling or detaching the spawned work, although note that
    /// once started, CPU work cannot be cancelled.
    pub fn spawn_cpu<F, R>(&self, f: F) -> Task<'scope, R>
    where
    // Unlike scheduling futures, the CPU task should have a static lifetime because it
    // doesn't need to access to handle to spawn more work.
        F: FnOnce() -> R + Send + 'scope,
        R: Send + 'scope,
    {
        let (send, recv) = oneshot::channel();

        let task = Box::new(move || {
            // Task::detach allows the receiver to be dropped, so we ignore send errors.
            let _ = send.send(f());
        });

        // Extend lifetime to 'rt for the runtime
        // SAFETY: Task<'scope, R> ensures it's awaited within 'scope
        let extended = unsafe {
            std::mem::transmute::<
                Box<dyn FnOnce() + Send + 'scope>,
                Box<dyn FnOnce() + Send + 'rt>,
            >(task)
        };

        let abort_handle = self.runtime.spawn_cpu(extended);
        Task {
            recv,
            abort_handle: Some(abort_handle),
        }
    }
}

impl<'rt> Handle<'rt, 'rt> {
    /// Open a file for I/O on this runtime.
    ///
    /// Since I/O tasks are processed by spawned background tasks on the runtime, we can only
    /// support opening files with the `'rt` lifetime. To give a counter-example, we open a file
    /// using a scoped handle, the scope ends and data is released, but the background tasks are
    /// still processing the request queue, potentially triggering a use-after-free.
    pub fn open_read<S: IntoIoSource>(&self, source: S) -> VortexResult<FileRead<'rt>> {
        let source = source.into_io_source()?;

        let (read, events) = FileRead::new(source.uri().clone(), source.size());

        let stream = IoRequestStream::new(
            StreamExt::boxed(events.to_async().into_stream()),
            source.coalesce_window(),
        )
        .boxed();

        self.runtime.spawn_io(IoTask::new(source, stream, self.clone()));

        Ok(read)
    }
}

/// A handle to a spawned Task.
///
/// If this handle is dropped, the task is cancelled where possible. In order to allow the task to
/// continue running in the background, call [`Task::detach`].
#[must_use = "When a Task is dropped without being awaited, it is cancelled"]
pub struct Task<'scope, T> {
    recv: oneshot::Receiver<T>,
    abort_handle: Option<Box<dyn AbortHandle + 'scope>>,
}

impl<'scope, T> Task<'scope, T> {
    /// Detach the task, allowing it to continue running in the background after being dropped.
    ///
    /// This is only possible if the task was spawned with a `'static` scope.
    pub fn detach(mut self) where 'scope: 'static {
        drop(self.abort_handle.take());
    }
}

impl<'scope, T> Future for Task<'scope, T> {
    type Output = T;

    fn poll(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match ready!(self.recv.poll_unpin(cx)) {
            Ok(result) => Poll::Ready(result),
            Err(recv_err) => {
                // If the other end of the channel was dropped, it means the runtime dropped
                // the future without ever completing it. If the caller aborted this task by
                // dropping it, then they wouldn't be able to poll it anymore.
                // So we consider a closed channel to be a Runtime programming error and therefore
                // we panic.
                vortex_panic!("Runtime dropped task without completing it: {recv_err}")
            }
        }
    }
}

impl<T> Drop for Task<'_, T> {
    fn drop(&mut self) {
        // Optimistically abort the task if it's still running.
        if let Some(handle) = self.abort_handle.take() {
            handle.abort();
        }
    }
}

#[cfg(test)]
mod test {
    use crate::runtime::single::SingleThreadRuntime;

    #[test]
    fn test_borrow_scope() {
        let a = 1;
        SingleThreadRuntime::block_on(|h| async {
            // We can spawn a future that borrows from the current scope
            assert_eq!(h.spawn(async {a + 1 }).await, 2);

            // But to borrow from _this_ closure, we need a nested scope
            let mut b = 2;
            h.scope(|h| {
                b += 1;
                h.spawn(async {
                    b += 1;
                })
            }).await;
            assert_eq!(b, 4);
        })
    }
}