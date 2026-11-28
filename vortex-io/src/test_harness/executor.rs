// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::future::Future;
use std::pin::pin;
use std::sync::Arc;
use std::sync::Weak;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::task::Context;
use std::task::Poll;
use std::task::Wake;
use std::task::Waker;

use futures::Stream;
use futures::StreamExt;
use futures::future::BoxFuture;
use futures::stream::BoxStream;
use parking_lot::Mutex;
use vortex_error::vortex_panic;

use crate::runtime::AbortHandle;
use crate::runtime::AbortHandleRef;
use crate::runtime::BlockingRuntime;
use crate::runtime::Executor;
use crate::runtime::Handle;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
struct TaskId(usize);

enum SimTask {
    Async(BoxFuture<'static, ()>),
    Sync(Box<dyn FnOnce() + Send + 'static>),
}

enum TaskSlot {
    Idle {
        task: SimTask,
        queued: bool,
    },
    Running {
        woken_during_poll: bool,
        aborted: bool,
    },
}

struct ExecutorState {
    next_id: usize,
    tasks: BTreeMap<TaskId, TaskSlot>,
    ready: VecDeque<TaskId>,
}

impl ExecutorState {
    fn new() -> Self {
        Self {
            next_id: 0,
            tasks: BTreeMap::new(),
            ready: VecDeque::new(),
        }
    }

    fn insert_task(&mut self, task: SimTask) -> TaskId {
        let id = TaskId(self.next_id);
        self.next_id += 1;
        self.tasks.insert(id, TaskSlot::Idle { task, queued: true });
        self.ready.push_back(id);
        id
    }

    fn pop_next_ready(&mut self) -> Option<(TaskId, SimTask)> {
        while let Some(id) = self.ready.pop_front() {
            let Some(slot) = self.tasks.get_mut(&id) else {
                continue;
            };
            if matches!(slot, TaskSlot::Running { .. }) {
                continue;
            }
            let prev = std::mem::replace(
                slot,
                TaskSlot::Running {
                    woken_during_poll: false,
                    aborted: false,
                },
            );
            if let TaskSlot::Idle { task, .. } = prev {
                return Some((id, task));
            }
        }
        None
    }
}

struct SimExecutorInner {
    state: Mutex<ExecutorState>,
}

impl Drop for SimExecutorInner {
    fn drop(&mut self) {
        // Drain all tasks without holding the mutex so dropping futures cannot deadlock.
        let tasks = std::mem::take(&mut self.state.get_mut().tasks);
        drop(tasks);
    }
}

impl SimExecutorInner {
    fn wake(&self, id: TaskId) {
        let mut state = self.state.lock();
        if let Some(slot) = state.tasks.get_mut(&id) {
            match slot {
                TaskSlot::Idle { queued, .. } => {
                    if !*queued {
                        *queued = true;
                        state.ready.push_back(id);
                    }
                }
                TaskSlot::Running {
                    woken_during_poll, ..
                } => {
                    *woken_during_poll = true;
                }
            }
        }
    }

    fn abort(&self, id: TaskId) {
        let dropped_task = {
            let mut state = self.state.lock();
            match state.tasks.get_mut(&id) {
                Some(TaskSlot::Idle { .. }) => match state.tasks.remove(&id) {
                    Some(TaskSlot::Idle { task, .. }) => Some(task),
                    _ => None,
                },
                Some(TaskSlot::Running { aborted, .. }) => {
                    *aborted = true;
                    None
                }
                None => None,
            }
        };
        // Drop the task outside the mutex in case its Drop impl interacts with the executor.
        drop(dropped_task);
    }

    fn spawn_task(self: &Arc<Self>, task: SimTask) -> AbortHandleRef {
        let id = self.state.lock().insert_task(task);
        Box::new(SimAbortHandle {
            inner: Arc::downgrade(self),
            id,
        })
    }

    fn step(self: &Arc<Self>) -> bool {
        let Some((id, task)) = self.state.lock().pop_next_ready() else {
            return false;
        };

        match task {
            SimTask::Sync(job) => {
                job();
                self.state.lock().tasks.remove(&id);
            }
            SimTask::Async(mut fut) => {
                let waker = Waker::from(Arc::new(SimWaker {
                    inner: Arc::downgrade(self),
                    id,
                }));
                let mut cx = Context::from_waker(&waker);
                let poll = fut.as_mut().poll(&mut cx);
                let to_drop = {
                    let mut state = self.state.lock();
                    match poll {
                        Poll::Ready(()) => {
                            state.tasks.remove(&id);
                            Some(fut)
                        }
                        Poll::Pending => match state.tasks.get(&id) {
                            Some(TaskSlot::Running {
                                woken_during_poll,
                                aborted,
                            }) => {
                                if *aborted {
                                    state.tasks.remove(&id);
                                    Some(fut)
                                } else {
                                    let woken = *woken_during_poll;
                                    state.tasks.insert(
                                        id,
                                        TaskSlot::Idle {
                                            task: SimTask::Async(fut),
                                            queued: woken,
                                        },
                                    );
                                    if woken {
                                        state.ready.push_back(id);
                                    }
                                    None
                                }
                            }
                            _ => Some(fut),
                        },
                    }
                };
                drop(to_drop);
            }
        }

        true
    }
}

struct ExecutorAdapter {
    inner: Arc<SimExecutorInner>,
}

impl Executor for ExecutorAdapter {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        self.inner.spawn_task(SimTask::Async(fut))
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        self.inner.spawn_task(SimTask::Sync(task))
    }

    fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        self.inner.spawn_task(SimTask::Sync(task))
    }
}

struct SimAbortHandle {
    inner: Weak<SimExecutorInner>,
    id: TaskId,
}

impl AbortHandle for SimAbortHandle {
    fn abort(self: Box<Self>) {
        if let Some(inner) = self.inner.upgrade() {
            inner.abort(self.id);
        }
    }
}

struct SimWaker {
    inner: Weak<SimExecutorInner>,
    id: TaskId,
}

impl Wake for SimWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        if let Some(inner) = self.inner.upgrade() {
            inner.wake(self.id);
        }
    }
}

struct RootWaker {
    woken: Arc<AtomicBool>,
}

impl Wake for RootWaker {
    fn wake(self: Arc<Self>) {
        self.wake_by_ref();
    }

    fn wake_by_ref(self: &Arc<Self>) {
        self.woken.store(true, Ordering::Release);
    }
}

/// Deterministic, single-threaded, manual-stepping FIFO executor for testing async workflows
/// without wall-clock sleeps or OS thread scheduling nondeterminism.
pub(crate) struct SimulatedExecutor {
    inner: Arc<SimExecutorInner>,
    adapter: Arc<dyn Executor>,
}

impl Default for SimulatedExecutor {
    fn default() -> Self {
        Self::new()
    }
}

impl SimulatedExecutor {
    /// Creates a deterministic FIFO executor.
    pub(crate) fn new() -> Self {
        let inner = Arc::new(SimExecutorInner {
            state: Mutex::new(ExecutorState::new()),
        });
        let adapter: Arc<dyn Executor> = Arc::new(ExecutorAdapter {
            inner: Arc::clone(&inner),
        });
        Self { inner, adapter }
    }

    /// Returns a Vortex runtime [`Handle`] targeting this simulated executor.
    #[expect(clippy::same_name_method)]
    pub(crate) fn handle(&self) -> Handle {
        Handle::new(Arc::downgrade(&self.adapter))
    }

    /// Drives all ready tasks until no runnable tasks remain, returning the number of steps run.
    pub(crate) fn run_until_stalled(&self) -> usize {
        let mut steps = 0;
        while self.inner.step() {
            steps += 1;
        }
        steps
    }

    /// Drives a root future alongside spawned tasks until the root future completes.
    ///
    /// Panics with `"SimulatedExecutor stalled ..."` if neither the root future nor any spawned
    /// task is runnable before `fut` resolves. Because `SimulatedExecutor` does not park on OS
    /// events, this also panics when forward progress depends on another OS thread (such as real
    /// asynchronous OS I/O or a Tokio runtime driving a future on a separate thread).
    pub(crate) fn drive<F: Future>(&self, fut: F) -> F::Output {
        let mut fut = pin!(fut);
        let root_woken = Arc::new(AtomicBool::new(true));
        let waker = Waker::from(Arc::new(RootWaker {
            woken: Arc::clone(&root_woken),
        }));
        let mut cx = Context::from_waker(&waker);

        loop {
            if root_woken.swap(false, Ordering::AcqRel)
                && let Poll::Ready(output) = fut.as_mut().poll(&mut cx)
            {
                return output;
            }

            if !self.inner.step() && !root_woken.load(Ordering::Acquire) {
                let pending = self.inner.state.lock().tasks.len();
                vortex_panic!(
                    "SimulatedExecutor stalled before root future completed ({pending} pending tasks)"
                );
            }
        }
    }
}

/// Iterator adapter driving a stream on a [`SimulatedExecutor`].
pub(crate) struct SimulatedIterator<'a, T> {
    executor: SimulatedExecutor,
    stream: BoxStream<'a, T>,
}

impl<T> Iterator for SimulatedIterator<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        self.executor.drive(self.stream.next())
    }
}

impl BlockingRuntime for SimulatedExecutor {
    type BlockingIterator<'a, R: 'a> = SimulatedIterator<'a, R>;

    fn handle(&self) -> Handle {
        SimulatedExecutor::handle(self)
    }

    fn block_on<Fut, R>(&self, fut: Fut) -> R
    where
        Fut: Future<Output = R>,
    {
        self.drive(fut)
    }

    fn block_on_stream<'a, S, R>(&self, stream: S) -> Self::BlockingIterator<'a, R>
    where
        S: Stream<Item = R> + Send + 'a,
        R: Send + 'a,
    {
        SimulatedIterator {
            executor: SimulatedExecutor {
                inner: Arc::clone(&self.inner),
                adapter: Arc::clone(&self.adapter),
            },
            stream: stream.boxed(),
        }
    }
}

#[cfg(test)]
mod tests {
    use futures::channel::oneshot;

    use super::*;

    #[test]
    #[should_panic(expected = "SimulatedExecutor stalled")]
    fn test_drive_panics_on_deadlock() {
        let exec = SimulatedExecutor::new();
        let (_tx, rx) = oneshot::channel::<()>();
        exec.drive(async move {
            let _ = rx.await;
        });
    }
}
