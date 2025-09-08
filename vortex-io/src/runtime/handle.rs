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
use crate::runtime::{AbortHandleRef, IoTask, Runtime};

/// A handle to an active Vortex runtime.
///
/// Users should obtain a handle from one of the runtime constructors and use it to spawn new
/// async tasks or CPU-heavy worker.
#[derive(Clone)]
pub struct Handle<'rt> {
    runtime: Arc<dyn Runtime>,
    _runtime: PhantomData<*mut &'rt ()>, // *mut makes it invariant which means it cannot be coerced
}

// Manually implement Send/Sync since *mut breaks auto-derive
// This is safe since runtime has `Send + Sync` bounds.
unsafe impl<'rt> Send for Handle<'rt> {}
unsafe impl<'rt> Sync for Handle<'rt> {}

impl<'rt> Handle<'rt> {
    pub(crate) fn new(runtime: Arc<dyn Runtime>) -> Self {
        Self {
            runtime,
            _runtime: PhantomData,
        }
    }

    /// Spawn a new future onto the runtime.
    ///
    /// These futures are expected to not perform expensive CPU work and instead simply schedule
    /// either CPU tasks or I/O tasks. See [`Handle::spawn_cpu`] for spawning CPU-bound work.
    ///
    /// See [`Task`] for details on cancelling or detaching the spawned task.
    pub fn spawn<Fut, R>(&self, f: Fut) -> Task<'rt, R>
    where
        Fut: Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        let (send, recv) = oneshot::channel();
        let abort_handle = self.runtime.spawn(
            async move {
                // Task::detach allows the receiver to be dropped, so we ignore send errors.
                let _ = send.send(f.await);
            }
            .boxed(),
        );
        Task {
            recv,
            abort_handle: Some(abort_handle),
            _runtime: PhantomData,
        }
    }

    /// Spawn a CPU-bound task for execution on the runtime.
    ///
    /// Note that many runtimes will interleave this work on the same async runtime. See the
    /// documentation for each runtime implementation for details.
    ///
    /// See [`Task`] for details on cancelling or detaching the spawned work, although note that
    /// once started, CPU work cannot be cancelled.
    pub fn spawn_cpu<F, R>(&self, f: F) -> Task<'rt, R>
    where
        // Unlike scheduling futures, the CPU task should have a static lifetime because it
        // doesn't need to access to handle to spawn more work.
        F: FnOnce() -> R + Send + 'static,
        R: Send + 'static,
    {
        // TODO(ngates): we want a droppable handle for this.
        let (send, recv) = oneshot::channel();
        let abort_handle = self.runtime.spawn_cpu(Box::new(move || {
            // Task::detach allows the receiver to be dropped, so we ignore send errors.
            let _ = send.send(f());
        }));
        Task {
            recv,
            abort_handle: Some(abort_handle),
            _runtime: PhantomData,
        }
    }

    /// Open a file for I/O on this runtime.
    pub fn open_read<S: IntoIoSource>(&self, source: S) -> VortexResult<FileRead<'rt>> {
        // The handle's lifetime is fake! The underlying runtime is Arc<dyn Runtime> which is
        // 'static. The reason we have a fake lifetime is to prevent users from passing back async
        // tasks outside the async context (i.e. after the runtime has been dropped) since they
        // may never be polled to completion, and it's a source of footgun bugs. However, there's
        // nothing unsafe about passing a 'static runtime to an I/O source.
        //
        // When we open I/O sources, we spawn a task to process their requests onto the runtime.
        // This task often needs to spawn _more_ tasks onto the runtime for parallelism. When a
        // `FileRead<'rt>` is dropped, the request channel will be closed and cause the I/O task
        // to eventually finish.
        let io_handle = Handle::new(self.runtime.clone());
        let source = source.into_io_source(io_handle)?;

        let (send, recv) = kanal::unbounded();

        let read = FileRead::new(source.uri().clone(), source.size(), send);

        let stream = IoRequestStream::new(
            StreamExt::boxed(recv.to_async().into_stream()),
            source.coalesce_window(),
        )
        .boxed();

        self.runtime.clone().spawn_io(IoTask::new(source, stream));

        Ok(read)
    }
}

/// A handle to a spawned Task.
///
/// If this handle is dropped, the task is cancelled where possible. In order to allow the task to
/// continue running in the background, call [`Task::detach`].
pub struct Task<'rt, T> {
    recv: oneshot::Receiver<T>,
    abort_handle: Option<AbortHandleRef>,
    _runtime: PhantomData<&'rt ()>,
}

impl<T> Task<'static, T> {
    /// Detach the task, allowing it to continue running in the background after being dropped.
    /// This is only possible if the underlying runtime has a 'static lifetime.
    pub fn detach(mut self) {
        drop(self.abort_handle.take());
    }
}

impl<'rt, T> Future for Task<'rt, T> {
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
