// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::rc::Rc;
use std::rc::Weak as RcWeak;
use std::sync::Arc;

use futures::Stream;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::LocalBoxStream;
use parking_lot::Mutex;
use smol::LocalExecutor;
use vortex_error::vortex_panic;

use crate::runtime::AbortHandle;
use crate::runtime::AbortHandleRef;
use crate::runtime::BlockingRuntime;
use crate::runtime::Executor;
use crate::runtime::Handle;
use crate::runtime::smol::SmolAbortHandle;

/// A runtime that drives all work on the current thread.
///
/// This is subtly different from using a current-thread runtime to drive a future since it is
/// capable of running `!Send` I/O futures.
pub struct SingleThreadRuntime {
    sender: Arc<Sender>,
    executor: Rc<LocalExecutor<'static>>,
}

impl Default for SingleThreadRuntime {
    fn default() -> Self {
        let executor = Rc::new(LocalExecutor::new());
        let sender = Arc::new(Sender::new(&executor));
        Self { sender, executor }
    }
}

struct Sender {
    scheduling: kanal::Sender<SpawnAsync<'static>>,
    cpu: kanal::Sender<SpawnSync<'static>>,
    blocking: kanal::Sender<SpawnSync<'static>>,
}

impl Sender {
    fn new(local: &Rc<LocalExecutor<'static>>) -> Self {
        let (scheduling_send, scheduling_recv) = kanal::unbounded::<SpawnAsync>();
        let (cpu_send, cpu_recv) = kanal::unbounded::<SpawnSync>();
        let (blocking_send, blocking_recv) = kanal::unbounded::<SpawnSync>();

        // We pass weak references to the local execution into the async tasks such that the task's
        // reference doesn't keep the execution alive after the runtime is dropped.
        let weak_local = Rc::downgrade(local);

        // Drive scheduling tasks.
        let weak_local2 = RcWeak::clone(&weak_local);
        local
            .spawn(async move {
                while let Ok(spawn) = scheduling_recv.as_async().recv().await {
                    if let Some(local) = weak_local2.upgrade() {
                        // Ignore send errors since it means the caller immediately detached.
                        drop(
                            spawn
                                .task_callback
                                .send(SmolAbortHandle::new_handle(local.spawn(spawn.future))),
                        );
                    }
                }
            })
            .detach();

        // Drive CPU tasks.
        let weak_local2 = RcWeak::clone(&weak_local);
        local
            .spawn(async move {
                while let Ok(spawn) = cpu_recv.as_async().recv().await {
                    if let Some(local) = weak_local2.upgrade() {
                        let work = spawn.sync;
                        // Ignore send errors since it means the caller immediately detached.
                        drop(spawn.task_callback.send(SmolAbortHandle::new_handle(
                            local.spawn(async move { work() }),
                        )));
                    }
                }
            })
            .detach();

        // Drive blocking tasks.
        let weak_local2 = RcWeak::clone(&weak_local);
        local
            .spawn(async move {
                while let Ok(spawn) = blocking_recv.as_async().recv().await {
                    if let Some(local) = weak_local2.upgrade() {
                        let work = spawn.sync;
                        // Ignore send errors since it means the caller immediately detached.
                        drop(spawn.task_callback.send(SmolAbortHandle::new_handle(
                            local.spawn(async move { work() }),
                        )));
                    }
                }
            })
            .detach();

        Self {
            scheduling: scheduling_send,
            cpu: cpu_send,
            blocking: blocking_send,
        }
    }
}

/// Since the [`Handle`], and therefore runtime implementation needs to be `Send` and `Sync`,
/// we cannot just `impl Runtime for LocalExecutor`. Instead, we create channels that the handle
/// can forward its work into, and we drive the resulting tasks on a [`LocalExecutor`] on the
/// calling thread.
impl Executor for Sender {
    fn spawn(&self, future: BoxFuture<'static, ()>) -> AbortHandleRef {
        let (send, recv) = oneshot::channel();
        if let Err(e) = self.scheduling.send(SpawnAsync {
            future,
            task_callback: send,
        }) {
            vortex_panic!("Executor missing: {}", e);
        }
        Box::new(LazyAbortHandle {
            task: Mutex::new(recv),
        })
    }

    fn spawn_cpu(&self, cpu: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        let (send, recv) = oneshot::channel();
        if let Err(e) = self.cpu.send(SpawnSync {
            sync: cpu,
            task_callback: send,
        }) {
            vortex_panic!("Executor missing: {}", e);
        }
        Box::new(LazyAbortHandle {
            task: Mutex::new(recv),
        })
    }

    fn spawn_blocking_io(&self, work: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        let (send, recv) = oneshot::channel();
        if let Err(e) = self.blocking.send(SpawnSync {
            sync: work,
            task_callback: send,
        }) {
            vortex_panic!("Executor missing: {}", e);
        }
        Box::new(LazyAbortHandle {
            task: Mutex::new(recv),
        })
    }
}

impl BlockingRuntime for SingleThreadRuntime {
    type BlockingIterator<'a, R: 'a> = SingleThreadIterator<'a, R>;

    fn handle(&self) -> Handle {
        let executor: Arc<dyn Executor> = Arc::clone(&self.sender) as Arc<dyn Executor>;
        Handle::new(Arc::downgrade(&executor))
    }

    fn block_on<Fut, R>(&self, fut: Fut) -> R
    where
        Fut: Future<Output = R>,
    {
        smol::block_on(self.executor.run(fut))
    }

    fn block_on_stream<'a, S, R>(&self, stream: S) -> Self::BlockingIterator<'a, R>
    where
        S: Stream<Item = R> + Send + 'a,
        R: Send + 'a,
    {
        SingleThreadIterator {
            executor: Rc::clone(&self.executor),
            stream: stream.boxed_local(),
        }
    }
}

/// Runs a future to completion on the current thread until it completes.
///
/// The future is provided a [`Handle`] to the runtime so that it may spawn additional tasks
/// to be executed concurrently.
pub fn block_on<F, Fut, R>(f: F) -> R
where
    F: FnOnce(Handle) -> Fut,
    Fut: Future<Output = R>,
{
    let runtime = SingleThreadRuntime::default();
    let handle = runtime.handle();
    runtime.block_on(f(handle))
}

/// Returns an iterator wrapper around a stream, blocking the current thread for each item.
pub fn block_on_stream<'a, F, S, R>(f: F) -> SingleThreadIterator<'a, R>
where
    F: FnOnce(Handle) -> S,
    S: Stream<Item = R> + Send + Unpin + 'a,
    R: Send + 'a,
{
    let runtime = SingleThreadRuntime::default();
    let handle = runtime.handle();
    runtime.block_on_stream(f(handle))
}

/// A spawn request for a future.
///
/// We pass back the abort handle via oneshot channel because this is a single-threaded runtime,
/// meaning we need the spawning channel consumer to do some work before the caller can actually
/// get ahold of their task handle.
///
/// The reason we don't pass back a smol::Task, and instead pass back a SmolAbortHandle, is because
/// we invert the behaviour of abort and drop. Dropping the abort handle results in the task being
/// detached, whereas dropping the smol::Task results in the task being canceled. This helps avoid
/// a race where the caller detaches the LazyAbortHandle before the smol::Task has been launched.
struct SpawnAsync<'rt> {
    future: BoxFuture<'rt, ()>,
    task_callback: oneshot::Sender<AbortHandleRef>,
}

// A spawn request for a synchronous job.
struct SpawnSync<'rt> {
    sync: Box<dyn FnOnce() + Send + 'rt>,
    task_callback: oneshot::Sender<AbortHandleRef>,
}

struct LazyAbortHandle {
    task: Mutex<oneshot::Receiver<AbortHandleRef>>,
}

impl AbortHandle for LazyAbortHandle {
    fn abort(self: Box<Self>) {
        // Aborting a smol::Task is done by dropping it.
        if let Ok(task) = self.task.lock().try_recv() {
            task.abort()
        }
    }
}

/// A stream that wraps up the stream with the execution that drives it.
pub struct SingleThreadIterator<'a, T> {
    executor: Rc<LocalExecutor<'static>>,
    stream: LocalBoxStream<'a, T>,
}

impl<T> Iterator for SingleThreadIterator<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        let fut = self.stream.next();
        smol::block_on(self.executor.run(fut))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::FutureExt;

    use crate::runtime::BlockingRuntime;
    use crate::runtime::single::SingleThreadRuntime;
    use crate::runtime::single::block_on;

    #[test]
    fn test_drive_simple_future() {
        let result = SingleThreadRuntime::default().block_on(async { 123 }.boxed_local());
        assert_eq!(result, 123);
    }

    #[test]
    fn test_spawn_cpu_task() {
        let counter = Arc::new(AtomicUsize::new(0));
        let c = Arc::clone(&counter);

        block_on(|handle| async move {
            handle
                .spawn_cpu(move || {
                    c.fetch_add(1, Ordering::SeqCst);
                })
                .await
        });

        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
