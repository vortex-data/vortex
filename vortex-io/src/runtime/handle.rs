// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::pin::Pin;
use std::sync::{Arc, Weak};
use std::task::{Context, Poll, ready};

use futures::channel::mpsc;
use futures::{FutureExt, StreamExt};
use vortex_error::{VortexResult, vortex_panic};
use vortex_metrics::VortexMetrics;

use crate::file::{FileRead, IntoReadSource, IoRequestStream};
use crate::runtime::{AbortHandleRef, Executor, IoTask};

/// A handle to an active Vortex runtime.
///
/// Users should obtain a handle from one of the Vortex runtime's and use it to spawn new async
/// tasks, blocking I/O tasks, CPU-heavy tasks, or to open files for reading or writing.
///
/// Note that a [`Handle`] is a weak reference to the underlying runtime. If the associated
/// runtime has been dropped, then any requests to spawn new tasks will panic.
#[derive(Clone)]
pub struct Handle {
    runtime: Weak<dyn Executor>,
}

impl Handle {
    pub(crate) fn new(runtime: Weak<dyn Executor>) -> Self {
        Self { runtime }
    }

    fn runtime(&self) -> Arc<dyn Executor> {
        self.runtime.upgrade().unwrap_or_else(|| {
            vortex_panic!("Attempted to use a Handle after its runtime was dropped")
        })
    }

    /// Returns a handle to the current runtime, if such a reasonable choice exists.
    ///
    /// For example, if called from within a Tokio context this will return a
    /// `TokioRuntime` handle.
    pub fn find() -> Option<Self> {
        #[cfg(feature = "tokio")]
        {
            use tokio::runtime::Handle as TokioHandle;
            if TokioHandle::try_current().is_ok() {
                return Some(crate::runtime::tokio::TokioRuntime::current());
            }
        }

        None
    }

    /// Spawn a new future onto the runtime.
    ///
    /// These futures are expected to not perform expensive CPU work and instead simply schedule
    /// either CPU tasks or I/O tasks. See [`Handle::spawn_cpu`] for spawning CPU-bound work.
    ///
    /// See [`Task`] for details on cancelling or detaching the spawned task.
    pub fn spawn<Fut, R>(&self, f: Fut) -> Task<R>
    where
        Fut: Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        let (send, recv) = oneshot::channel();
        let abort_handle = self.runtime().spawn(
            async move {
                // Task::detach allows the receiver to be dropped, so we ignore send errors.
                drop(send.send(f.await));
            }
            .boxed(),
        );
        Task {
            recv,
            abort_handle: Some(abort_handle),
        }
    }

    /// A helper function to avoid manually cloning the handle when spawning nested tasks.
    pub fn spawn_nested<F, Fut, R>(&self, f: F) -> Task<R>
    where
        F: FnOnce(Handle) -> Fut,
        Fut: Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        self.spawn(f(Handle::new(self.runtime.clone())))
    }

    /// Spawn a CPU-bound task for execution on the runtime.
    ///
    /// Note that many runtimes will interleave this work on the same async runtime. See the
    /// documentation for each runtime implementation for details.
    ///
    /// See [`Task`] for details on cancelling or detaching the spawned work, although note that
    /// once started, CPU work cannot be cancelled.
    pub fn spawn_cpu<F, R>(&self, f: F) -> Task<R>
    where
        // Unlike scheduling futures, the CPU task should have a static lifetime because it
        // doesn't need to access to handle to spawn more work.
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (send, recv) = oneshot::channel();
        let abort_handle = self.runtime().spawn_cpu(Box::new(move || {
            // Optimistically avoid the work if the result won't be used.
            if !send.is_closed() {
                // Task::detach allows the receiver to be dropped, so we ignore send errors.
                drop(send.send(f()));
            }
        }));
        Task {
            recv,
            abort_handle: Some(abort_handle),
        }
    }

    /// Spawn a blocking I/O task for execution on the runtime.
    pub fn spawn_blocking<F, R>(&self, f: F) -> Task<R>
    where
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        let (send, recv) = oneshot::channel();
        let abort_handle = self.runtime().spawn_blocking(Box::new(move || {
            // Optimistically avoid the work if the result won't be used.
            if !send.is_closed() {
                // Task::detach allows the receiver to be dropped, so we ignore send errors.
                drop(send.send(f()));
            }
        }));
        Task {
            recv,
            abort_handle: Some(abort_handle),
        }
    }

    /// Open a file for I/O on this runtime.
    pub fn open_read<S: IntoReadSource>(
        &self,
        source: S,
        metrics: VortexMetrics,
    ) -> VortexResult<FileRead> {
        let source = source.into_read_source(self.clone())?;

        let (send, recv) = mpsc::unbounded();

        let read = FileRead::new(source.uri().clone(), source.size(), send);

        let stream =
            IoRequestStream::new(StreamExt::boxed(recv), source.coalesce_window(), metrics).boxed();

        self.runtime().spawn_io(IoTask::new(source, stream));

        Ok(read)
    }
}

/// A handle to a spawned Task.
///
/// If this handle is dropped, the task is cancelled where possible. In order to allow the task to
/// continue running in the background, call [`Task::detach`].
#[must_use = "When a Task is dropped without being awaited, it is cancelled"]
pub struct Task<T> {
    recv: oneshot::Receiver<T>,
    abort_handle: Option<AbortHandleRef>,
}

impl<T> Task<T> {
    /// Detach the task, allowing it to continue running in the background after being dropped.
    /// This is only possible if the underlying runtime has a 'static lifetime.
    pub fn detach(mut self) {
        drop(self.abort_handle.take());
    }
}

impl<T> Future for Task<T> {
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
                vortex_panic!(
                    "Runtime dropped task without completing it, likely it panicked: {recv_err}"
                )
            }
        }
    }
}

impl<T> Drop for Task<T> {
    fn drop(&mut self) {
        // Optimistically abort the task if it's still running.
        if let Some(handle) = self.abort_handle.take() {
            handle.abort();
        }
    }
}
