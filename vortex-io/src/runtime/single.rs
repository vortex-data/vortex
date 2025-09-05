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
pub struct SingleThreadRuntime<'rt> {
    scheduling: kanal::Sender<SpawnFuture<'rt>>,
    cpu: kanal::Sender<SpawnCpu>,
    io: kanal::Sender<IoTask<'rt>>,
}

impl<'rt> SingleThreadRuntime<'rt> {
    fn new<'ex>() -> (Self, Rc<LocalExecutor<'ex>>)
    where
        'rt: 'ex,
    {
        let (scheduling_send, scheduling_recv) = kanal::unbounded::<SpawnFuture<'rt>>();
        let (cpu_send, cpu_recv) = kanal::unbounded::<SpawnCpu>();
        let (io_send, io_recv) = kanal::unbounded::<IoTask>();

        let local = Rc::new(LocalExecutor::new());

        // We pass weak references to the local executor into the async tasks such that the task's
        // reference doesn't keep the executor alive after the runtime is dropped.
        let weak_local = Rc::downgrade(&local);

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
                        local.spawn(task.drive_local()).detach();
                    }
                }
            })
            .detach();

        (
            Self {
                scheduling: scheduling_send,
                cpu: cpu_send,
                io: io_send,
            },
            local,
        )
    }

    /// Drive the given Vortex future on the underlying single-threaded runtime.
    pub fn block_on<'fut, F, Fut, R>(f: F) -> R
    where
        F: FnOnce(Handle<'rt>) -> Fut,
        Fut: Future<Output = R> + 'fut,
        R: Send + 'static,
    {
        let (rt, executor) = SingleThreadRuntime::new();
        let fut = f(Handle(Arc::new(rt)));
        block_on(executor.run(fut))
    }

    /// Drive the given Vortex stream on the underlying single-threaded runtime.
    pub fn block_on_stream<F, S, R>(f: F) -> impl Iterator<Item = R>
    where
        F: FnOnce(Handle<'rt>) -> S,
        S: Stream<Item = R> + Unpin,
        R: Send + 'static,
    {
        // Create a new static executor.
        let (rt, executor) = SingleThreadRuntime::new();
        let stream = f(Handle(Arc::new(rt)));

        // SAFETY: The stream contains references to `rt` with lifetime 'rt.
        // We're transmuting this to static, which is sound because:
        // 1. Both `rt` and `stream` will be moved into BlockingStream
        // 2. BlockingStream will drop them in the correct order (stream first, then rt)
        // 3. The stream will never outlive the runtime it references
        let stream: LocalBoxStream<'static, R> = unsafe {
            std::mem::transmute::<LocalBoxStream<'_, R>, LocalBoxStream<'static, R>>(
                stream.boxed_local(),
            )
        };
        let executor: Rc<LocalExecutor<'static>> = unsafe {
            std::mem::transmute::<Rc<LocalExecutor<'_>>, Rc<LocalExecutor<'static>>>(executor)
        };

        BlockingStream { executor, stream }
    }
}

/// Since the [`Handle`], and therefore runtime implementation needs to be `Send` and `Sync`,
/// we cannot just `impl Runtime for LocalExecutor`. Instead, we create channels that the handle
/// can forward its work into, and we drive the resulting tasks on a [`LocalExecutor`] on the
/// calling thread.
impl<'rt> Runtime<'rt> for SingleThreadRuntime<'rt> {
    fn spawn(&self, future: BoxFuture<'rt, ()>) -> AbortHandleRef<'rt> {
        let task = Arc::new(Mutex::new(None));
        if let Err(e) = self.scheduling.send(SpawnFuture {
            future,
            task: task.clone(),
        }) {
            vortex_panic!("Executor missing: {}", e);
        }
        Box::new(SmolAbortHandle { task })
    }

    fn spawn_cpu(&self, cpu: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef<'rt> {
        let task = Arc::new(Mutex::new(None));
        if let Err(e) = self.cpu.send(SpawnCpu {
            cpu,
            task: task.clone(),
        }) {
            vortex_panic!("Executor missing: {}", e);
        }
        Box::new(SmolAbortHandle { task })
    }

    fn spawn_io(&self, task: IoTask<'rt>) {
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
struct SpawnCpu {
    cpu: Box<dyn FnOnce() + Send + 'static>,
    task: Arc<Mutex<Option<smol::Task<()>>>>,
}

struct SmolAbortHandle {
    task: Arc<Mutex<Option<smol::Task<()>>>>,
}

impl<'rt> AbortHandle<'rt> for SmolAbortHandle {
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
///
/// This allows the resulting stream to have a static lifetime.
struct BlockingStream<T> {
    executor: Rc<LocalExecutor<'static>>,
    stream: LocalBoxStream<'static, T>,
}

impl<T> Iterator for BlockingStream<T> {
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

    use super::*;

    #[test]
    fn test_drive_simple_future() {
        let result = SingleThreadRuntime::block_on(|_handle| async { 123 });
        assert_eq!(result, 123);
    }

    #[test]
    fn test_spawn_cpu_task() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = counter.clone();

        let result = SingleThreadRuntime::block_on(|handle| async move {
            handle
                .spawn_cpu(move || c.fetch_add(1, Ordering::SeqCst))
                .await
        });

        assert_eq!(result, 0);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
