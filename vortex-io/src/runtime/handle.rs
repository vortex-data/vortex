// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::panic::AssertUnwindSafe;
use std::pin::Pin;
use std::sync::Arc;
use std::sync::Weak;
use std::task::Context;
use std::task::Poll;
use std::task::ready;

use futures::FutureExt;
use futures::channel::oneshot;
use tracing::Instrument;
use vortex_error::vortex_panic;

use crate::runtime::AbortHandleRef;
use crate::runtime::Executor;
use crate::runtime::platform;

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
    pub fn new(runtime: Weak<dyn Executor>) -> Self {
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
    /// `TokioRuntime` handle. On browser WebAssembly with the `wasm-bindgen` feature this returns
    /// the global `WasmRuntime` handle; without it there is no event loop to schedule onto, and
    /// callers must drive a `SingleThreadRuntime` and install its handle themselves.
    pub fn find() -> Option<Self> {
        #[cfg(feature = "tokio")]
        {
            use tokio::runtime::Handle as TokioHandle;

            use crate::runtime::tokio::TokioRuntime;
            if TokioHandle::try_current().is_ok() {
                return Some(TokioRuntime::current());
            }
        }

        platform::default_handle()
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
        // Instrument with a dedicated, named span on its own target rather than
        // re-entering `Span::current()`. `Instrumented::poll` enters and exits the
        // span on every poll, so re-entering the caller's span makes its cost scale
        // with poll count. A distinct target lets subscribers opt these spans in or
        // out via filtering; when filtered out the span is disabled and enter/exit
        // are no-ops, which keeps frequently-polled spawned futures cheap.
        let span = tracing::trace_span!(target: "vortex_io::spawn", "spawn");
        let abort_handle = self.runtime().spawn(
            async move {
                // Catch a panic so it can be re-raised on the joining side (see `Task::poll`)
                // rather than being lost, matching `tokio::JoinError` / `smol::Task` semantics.
                let output = AssertUnwindSafe(f).catch_unwind().await;
                // Task::detach allows the receiver to be dropped, so we ignore send errors.
                drop(send.send(output));
            }
            .instrument(span)
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
        self.spawn(f(Handle::new(Weak::clone(&self.runtime))))
    }

    /// Spawn a new I/O future onto the runtime.
    ///
    /// See [`Executor::spawn_io`] for more details about how this future is expected to run.
    // See [`Task`] for details on cancelling or detaching the spawned task.
    pub fn spawn_io<Fut, R>(&self, f: Fut) -> Task<R>
    where
        Fut: Future<Output = R> + Send + 'static,
        R: Send + 'static,
    {
        let (send, recv) = oneshot::channel();
        // See `spawn` above: a dedicated target rather than `Span::current()` so
        // subscribers can filter these spans in or out. I/O futures are polled
        // frequently, so disabling the span (the default for an unconfigured
        // target) avoids per-poll enter/exit cost.
        let span = tracing::trace_span!(target: "vortex_io::spawn_io", "spawn_io");
        let abort_handle = self.runtime().spawn_io(
            async move {
                // See `spawn`: catch a panic so it re-raises on the joining side.
                let output = AssertUnwindSafe(f).catch_unwind().await;
                // Task::detach allows the receiver to be dropped, so we ignore send errors.
                drop(send.send(output));
            }
            .instrument(span)
            .boxed(),
        );
        Task {
            recv,
            abort_handle: Some(abort_handle),
        }
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
        let span = tracing::Span::current();
        let abort_handle = self.runtime().spawn_cpu(Box::new(move || {
            let _guard = span.enter();
            // Optimistically avoid the work if the result won't be used.
            if !send.is_canceled() {
                // Catch a panic so it re-raises on the joining side (see `Task::poll`).
                let output = std::panic::catch_unwind(AssertUnwindSafe(f));
                // Task::detach allows the receiver to be dropped, so we ignore send errors.
                drop(send.send(output));
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
        let span = tracing::Span::current();
        let abort_handle = self.runtime().spawn_blocking_io(Box::new(move || {
            let _guard = span.enter();
            // Optimistically avoid the work if the result won't be used.
            if !send.is_canceled() {
                // Catch a panic so it re-raises on the joining side (see `Task::poll`).
                let output = std::panic::catch_unwind(AssertUnwindSafe(f));
                // Task::detach allows the receiver to be dropped, so we ignore send errors.
                drop(send.send(output));
            }
        }));
        Task {
            recv,
            abort_handle: Some(abort_handle),
        }
    }
}

/// The value carried from a spawned task back to its [`Task`] handle: either the task's output,
/// or the panic payload if the task panicked, so it can be re-raised on the joining side.
type TaskOutput<T> = Result<T, Box<dyn Any + Send>>;

/// The terminal outcome of joining a spawned [`Task`] with [`Task::poll_join`], without
/// re-raising a panic.
pub enum JoinOutcome<T> {
    /// The task ran to completion and produced this value.
    Completed(T),
    /// The task panicked. Carries the original panic payload, ready to be re-raised with
    /// [`std::panic::resume_unwind`].
    Panicked(Box<dyn Any + Send>),
    /// The runtime dropped the task's future before it completed, for example because the task
    /// was aborted or the runtime shut down. No value or panic payload is available.
    Aborted,
}

/// A handle to a spawned Task.
///
/// If this handle is dropped, the task is cancelled where possible. In order to allow the task to
/// continue running in the background, call [`Task::detach`].
#[must_use = "When a Task is dropped without being awaited, it is cancelled"]
pub struct Task<T> {
    recv: oneshot::Receiver<TaskOutput<T>>,
    abort_handle: Option<AbortHandleRef>,
}

impl<T> Task<T> {
    /// Detach the task, allowing it to continue running in the background after being dropped.
    /// This is only possible if the underlying runtime has a 'static lifetime.
    pub fn detach(mut self) {
        drop(self.abort_handle.take());
    }

    /// Poll the task to completion, reporting the terminal state as a [`JoinOutcome`] instead of
    /// re-raising a panic or panicking on abort.
    ///
    /// This lets a caller distinguish a task panic (which it may re-raise via
    /// [`std::panic::resume_unwind`]) from the runtime dropping the task before it completed — for
    /// example a runtime shutting down while work is in flight — which is benign. The [`Future`]
    /// implementation, by contrast, re-raises a panic and itself panics on abort.
    pub fn poll_join(&mut self, cx: &mut Context<'_>) -> Poll<JoinOutcome<T>> {
        match ready!(self.recv.poll_unpin(cx)) {
            Ok(Ok(output)) => Poll::Ready(JoinOutcome::Completed(output)),
            Ok(Err(panic)) => Poll::Ready(JoinOutcome::Panicked(panic)),
            // The result channel closed without delivering a value: the runtime dropped the task's
            // future before it completed (for example the task was aborted or the runtime shut
            // down).
            Err(_recv_err) => Poll::Ready(JoinOutcome::Aborted),
        }
    }
}

impl<T> Future for Task<T> {
    type Output = T;

    #[expect(clippy::panic)]
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        match ready!(self.get_mut().poll_join(cx)) {
            JoinOutcome::Completed(output) => Poll::Ready(output),
            // The task panicked: re-raise the original panic on the joining side, preserving its
            // message and payload rather than collapsing it into a generic error.
            JoinOutcome::Panicked(panic) => std::panic::resume_unwind(panic),
            JoinOutcome::Aborted => {
                // The runtime dropped the task's future before it completed. If the caller aborted
                // this task by dropping it, they wouldn't be able to poll it anymore, so we
                // consider a closed channel a runtime programming error and panic.

                // NOTE(ngates): we don't use vortex_panic to avoid printing a useless backtrace.
                panic!("Runtime dropped task without completing it")
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

#[cfg(test)]
mod tests {
    use futures::task::noop_waker;

    use super::*;

    // A task whose result channel closed without a value — the runtime dropped its future before
    // it sent a result — must report `Aborted`, so callers can treat a benign teardown as a
    // recoverable stop rather than a propagated panic.
    #[test]
    fn poll_join_reports_aborted_when_channel_closed_without_value() {
        let (send, recv) = oneshot::channel::<TaskOutput<()>>();
        // Simulate the runtime dropping the task's future before it sent a result.
        drop(send);

        let mut task = Task::<()> {
            recv,
            abort_handle: None,
        };

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        assert!(matches!(
            task.poll_join(&mut cx),
            Poll::Ready(JoinOutcome::Aborted)
        ));
    }

    // A completed task reports its value through `poll_join` rather than re-raising or panicking.
    #[test]
    fn poll_join_reports_completed_value() {
        let (send, recv) = oneshot::channel::<TaskOutput<u32>>();
        // Ignore the send result: the payload type is not `Debug`, and the receiver is alive.
        drop(send.send(Ok(7)));

        let mut task = Task::<u32> {
            recv,
            abort_handle: None,
        };

        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        assert!(matches!(
            task.poll_join(&mut cx),
            Poll::Ready(JoinOutcome::Completed(7))
        ));
    }
}
