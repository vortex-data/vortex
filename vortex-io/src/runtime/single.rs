// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::rc::Rc;
use std::sync::Arc;

use futures::future::BoxFuture;
use futures::stream::LocalBoxStream;
use futures::{Stream, StreamExt};
use parking_lot::Mutex;
use smol::{LocalExecutor, block_on};
use vortex_error::vortex_panic;

use crate::runtime::{AbortHandle, AbortHandleRef, Handle, IoTask, Runtime};

/// A runtime that drives all work on the current thread.
///
/// This is subtly different from using a current-thread runtime to drive a future since it is
/// capable of running `!Send` I/O futures.
pub struct SingleThreadRuntime {
    scheduling: kanal::Sender<SpawnFuture<'static>>,
    cpu: kanal::Sender<SpawnCpu<'static>>,
    io: kanal::Sender<IoTask>,
}

impl SingleThreadRuntime {
    fn new(local: &Rc<LocalExecutor<'static>>) -> Self {
        let (scheduling_send, scheduling_recv) = kanal::unbounded::<SpawnFuture>();
        let (cpu_send, cpu_recv) = kanal::unbounded::<SpawnCpu>();
        let (io_send, io_recv) = kanal::unbounded::<IoTask>();

        // We pass weak references to the local executor into the async tasks such that the task's
        // reference doesn't keep the executor alive after the runtime is dropped.
        let weak_local = Rc::downgrade(local);

        // Drive scheduling tasks.
        let weak_local2 = weak_local.clone();
        local
            .spawn(async move {
                while let Ok(spawn) = scheduling_recv.as_async().recv().await {
                    if let Some(local) = weak_local2.upgrade() {
                        *spawn.task.lock() = Some(local.spawn(spawn.future));
                    }
                }
            })
            .detach();

        // Drive CPU tasks.
        let weak_local2 = weak_local.clone();
        local
            .spawn(async move {
                while let Ok(spawn) = cpu_recv.as_async().recv().await {
                    if let Some(local) = weak_local2.upgrade() {
                        let cpu = spawn.cpu;
                        *spawn.task.lock() = Some(local.spawn(async move { cpu() }));
                    }
                }
            })
            .detach();

        // Drive I/O tasks.
        let weak_local2 = weak_local;
        local
            .spawn(async move {
                while let Ok(task) = io_recv.as_async().recv().await {
                    if let Some(local) = weak_local2.upgrade() {
                        local.spawn(task.source.drive_local(task.stream)).detach();
                    }
                }
            })
            .detach();

        Self {
            scheduling: scheduling_send,
            cpu: cpu_send,
            io: io_send,
        }
    }

    /// Drive the given Vortex future on the underlying single-threaded runtime.
    pub fn block_on<F, Fut, R>(f: F) -> R
    where
        F: FnOnce(Handle) -> Fut,
        Fut: Future<Output = R>,
    {
        let executor = Rc::new(LocalExecutor::new());
        let runtime = Arc::new(SingleThreadRuntime::new(&executor));
        let handle = Handle::new(runtime);

        let fut = f(handle);
        block_on(executor.run(fut))
    }

    /// Drive the given Vortex stream on the underlying single-threaded runtime.
    pub fn block_on_stream<'a, F, S, R>(f: F) -> impl Iterator<Item = R> + 'a
    where
        F: FnOnce(Handle) -> S,
        S: Stream<Item = R> + 'a,
        R: 'a,
    {
        let executor = Rc::new(LocalExecutor::new());
        let handle = Handle::new(Arc::new(Self::new(&executor)));
        let stream = f(handle).boxed_local();
        BlockingStream { executor, stream }
    }
}

/// Since the [`Handle`], and therefore runtime implementation needs to be `Send` and `Sync`,
/// we cannot just `impl Runtime for LocalExecutor`. Instead, we create channels that the handle
/// can forward its work into, and we drive the resulting tasks on a [`LocalExecutor`] on the
/// calling thread.
impl Runtime for SingleThreadRuntime {
    fn spawn(&self, future: BoxFuture<'static, ()>) -> AbortHandleRef {
        let task = Arc::new(Mutex::new(None));
        if let Err(e) = self.scheduling.send(SpawnFuture {
            future,
            task: task.clone(),
        }) {
            vortex_panic!("Executor missing: {}", e);
        }
        Box::new(SmolAbortHandle { task })
    }

    fn spawn_cpu(&self, cpu: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        let task = Arc::new(Mutex::new(None));
        if let Err(e) = self.cpu.send(SpawnCpu {
            cpu,
            task: task.clone(),
        }) {
            vortex_panic!("Executor missing: {}", e);
        }
        Box::new(SmolAbortHandle { task })
    }

    fn spawn_io(&self, task: IoTask) {
        if let Err(e) = self.io.send(task) {
            vortex_panic!("Executor missing: {}", e);
        }
    }
}

// A spawn request for a future.
struct SpawnFuture<'rt> {
    future: BoxFuture<'rt, ()>,
    task: Arc<Mutex<Option<smol::Task<()>>>>,
}

// A spawn request for a CPU job.
struct SpawnCpu<'rt> {
    cpu: Box<dyn FnOnce() + Send + 'rt>,
    task: Arc<Mutex<Option<smol::Task<()>>>>,
}

struct SmolAbortHandle {
    task: Arc<Mutex<Option<smol::Task<()>>>>,
}

impl AbortHandle for SmolAbortHandle {
    fn abort(self: Box<Self>) {
        // Aborting a smol::Task is done by dropping it.
        if let Some(task) = self.task.lock().take() {
            drop(task);
        }
    }
}

impl Drop for SmolAbortHandle {
    fn drop(&mut self) {
        // We prevent the task from being canceled by detaching it.
        if let Some(task) = self.task.lock().take() {
            task.detach();
        }
    }
}

/// A stream that wraps up the stream with the executor that drives it.
struct BlockingStream<'a, T> {
    executor: Rc<LocalExecutor<'static>>,
    stream: LocalBoxStream<'a, T>,
}

impl<T> Iterator for BlockingStream<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let fut = self.stream.next();
        block_on(self.executor.run(fut))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use futures::FutureExt;

    use crate::runtime::single::SingleThreadRuntime;

    #[test]
    fn test_drive_simple_future() {
        let result = SingleThreadRuntime::block_on(|_handle| async { 123 }.boxed_local());
        assert_eq!(result, 123);
    }

    #[test]
    fn test_spawn_cpu_task() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();

        SingleThreadRuntime::block_on(|handle| async move {
            handle
                .spawn_cpu(move || {
                    c.fetch_add(1, Ordering::SeqCst);
                })
                .await
        });

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    /// Returning a future that references the handle is not allowed, e.g.
    ///
    /// ```compile_fail
    /// use vortex_io::runtime::single::SingleThreadRuntime;
    /// use futures::FutureExt;
    ///
    /// SingleThreadRuntime::block_on(|handle| async move {
    ///     handle.spawn_cpu(move || 123)
    /// })
    /// ```
    #[test]
    fn test_handle_scope() {
        // But returning a result that _doesn't_ reference the handle is allowed.
        let result = SingleThreadRuntime::block_on(|handle| {
            async move { handle.spawn_cpu(move || 123).await }.boxed_local()
        });
        assert_eq!(result, 123);

        // E.g. this should not compile:
        // let result =
        //     SingleThreadRuntime::block_on(|handle| async move { handle.spawn_cpu(move || 123) });
    }
}
