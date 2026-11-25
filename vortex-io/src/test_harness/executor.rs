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
use crate::test_harness::DeterministicRng;

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
    rng: Option<DeterministicRng>,
}

impl ExecutorState {
    fn new(rng: Option<DeterministicRng>) -> Self {
        Self {
            next_id: 0,
            tasks: BTreeMap::new(),
            ready: VecDeque::new(),
            rng,
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
        while !self.ready.is_empty() {
            let id = match self.rng.as_mut() {
                Some(rng) => {
                    let idx = rng.next_usize(self.ready.len());
                    self.ready.swap_remove_back(idx)
                }
                None => self.ready.pop_front(),
            }?;

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

/// Deterministic, single-threaded, manual-stepping executor for testing async workflows
/// without wall-clock sleeps or OS thread scheduling nondeterminism.
///
/// Supports both FIFO task polling ([`SimulatedExecutor::new`]) and seeded deterministic
/// scheduling ([`SimulatedExecutor::with_seed`]).
pub struct SimulatedExecutor {
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
    pub fn new() -> Self {
        let inner = Arc::new(SimExecutorInner {
            state: Mutex::new(ExecutorState::new(None)),
        });
        let adapter: Arc<dyn Executor> = Arc::new(ExecutorAdapter {
            inner: Arc::clone(&inner),
        });
        Self { inner, adapter }
    }

    /// Creates a deterministic executor that picks among ready tasks using a seeded PRNG.
    pub fn with_seed(seed: u64) -> Self {
        let inner = Arc::new(SimExecutorInner {
            state: Mutex::new(ExecutorState::new(Some(DeterministicRng::new(seed)))),
        });
        let adapter: Arc<dyn Executor> = Arc::new(ExecutorAdapter {
            inner: Arc::clone(&inner),
        });
        Self { inner, adapter }
    }

    /// Returns a Vortex runtime [`Handle`] targeting this simulated executor.
    #[expect(clippy::same_name_method)]
    pub fn handle(&self) -> Handle {
        Handle::new(Arc::downgrade(&self.adapter))
    }

    /// Polls or executes a single ready task. Returns `false` if no tasks are currently runnable.
    pub fn step(&self) -> bool {
        self.inner.step()
    }

    /// Drives all ready tasks until no runnable tasks remain, returning the number of steps run.
    pub fn run_until_stalled(&self) -> usize {
        let mut steps = 0;
        while self.step() {
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
    pub fn drive<F: Future>(&self, fut: F) -> F::Output {
        let mut fut = pin!(fut);
        let root_woken = Arc::new(AtomicBool::new(true));
        let waker = Waker::from(Arc::new(RootWaker {
            woken: Arc::clone(&root_woken),
        }));
        let mut cx = Context::from_waker(&waker);
        let mut first_poll = true;

        loop {
            let prefer_spawned = !first_poll && {
                let mut state = self.inner.state.lock();
                !state.ready.is_empty()
                    && state.rng.as_mut().is_some_and(|rng| rng.next_usize(2) == 0)
            };

            if prefer_spawned && self.step() {
                continue;
            }

            if root_woken.swap(false, Ordering::AcqRel) {
                first_poll = false;
                if let Poll::Ready(output) = fut.as_mut().poll(&mut cx) {
                    return output;
                }
            }

            if !self.step() && !root_woken.load(Ordering::Acquire) {
                vortex_panic!(
                    "SimulatedExecutor stalled before root future completed ({} pending tasks)",
                    self.pending_task_count()
                );
            }
        }
    }

    /// Number of tasks that have been spawned and have not yet completed or been aborted.
    pub fn pending_task_count(&self) -> usize {
        self.inner.state.lock().tasks.len()
    }
}

/// Iterator adapter driving a stream on a [`SimulatedExecutor`].
pub struct SimulatedIterator<'a, T> {
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
    use std::sync::Arc;
    use std::sync::atomic::AtomicBool;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use futures::channel::oneshot;
    use futures::stream;
    use parking_lot::Mutex;

    use super::*;

    struct DropFlag(Arc<AtomicBool>);
    impl Drop for DropFlag {
        fn drop(&mut self) {
            self.0.store(true, Ordering::SeqCst);
        }
    }

    #[test]
    fn test_fifo_step_and_run_until_stalled() {
        let exec = SimulatedExecutor::new();
        let order = Arc::new(Mutex::new(Vec::new()));

        let mut tasks = Vec::new();
        for i in 0..3 {
            let order = Arc::clone(&order);
            tasks.push(exec.handle().spawn(async move {
                order.lock().push(i);
            }));
        }

        assert_eq!(exec.pending_task_count(), 3);
        assert!(exec.step());
        assert_eq!(*order.lock(), vec![0]);

        let remaining = exec.run_until_stalled();
        assert_eq!(remaining, 2);
        assert_eq!(*order.lock(), vec![0, 1, 2]);
        assert_eq!(exec.pending_task_count(), 0);
        drop(tasks);
    }

    #[test]
    fn test_seeded_scheduling_is_deterministic() {
        let run = |seed: u64| {
            let exec = SimulatedExecutor::with_seed(seed);
            let order = Arc::new(Mutex::new(Vec::new()));
            let mut tasks = Vec::new();
            for i in 0..8 {
                let order = Arc::clone(&order);
                tasks.push(exec.handle().spawn(async move {
                    order.lock().push(i);
                }));
            }
            exec.run_until_stalled();
            drop(tasks);
            order.lock().clone()
        };

        assert_eq!(run(42), run(42));
        assert_ne!(run(42), run(43));
    }

    #[test]
    fn test_abort_drops_pending_future_and_sync_tasks() {
        let exec = SimulatedExecutor::new();
        let dropped = Arc::new(AtomicBool::new(false));
        let flag = DropFlag(Arc::clone(&dropped));
        let (_tx, rx) = oneshot::channel::<()>();

        let async_task = exec.handle().spawn(async move {
            let _guard = flag;
            let _ = rx.await;
        });
        assert_eq!(exec.run_until_stalled(), 1);
        assert!(!dropped.load(Ordering::SeqCst));

        drop(async_task);
        assert!(dropped.load(Ordering::SeqCst));
        assert_eq!(exec.pending_task_count(), 0);

        let ran = Arc::new(AtomicUsize::new(0));
        let ran_clone = Arc::clone(&ran);
        let cpu_task = exec.handle().spawn_cpu(move || {
            ran_clone.fetch_add(1, Ordering::SeqCst);
        });
        drop(cpu_task);
        exec.run_until_stalled();
        assert_eq!(ran.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_detached_task_runs_to_completion_after_gate_opens() {
        let exec = SimulatedExecutor::new();
        let (gate_tx, gate_rx) = oneshot::channel::<()>();
        let done = Arc::new(AtomicBool::new(false));
        let done_clone = Arc::clone(&done);

        exec.handle()
            .spawn(async move {
                let _ = gate_rx.await;
                done_clone.store(true, Ordering::SeqCst);
            })
            .detach();

        exec.run_until_stalled();
        assert!(!done.load(Ordering::SeqCst));

        assert!(gate_tx.send(()).is_ok());
        exec.run_until_stalled();
        assert!(done.load(Ordering::SeqCst));
        assert_eq!(exec.pending_task_count(), 0);
    }

    #[test]
    fn test_blocking_runtime_block_on_and_stream() {
        let exec = SimulatedExecutor::new();
        let value = exec.block_on(async {
            let a = exec.handle().spawn_cpu(|| 20).await;
            let b = exec.handle().spawn_blocking(|| 22).await;
            a + b
        });
        assert_eq!(value, 42);

        let items: Vec<_> = exec.block_on_stream(stream::iter([1, 2, 3])).collect();
        assert_eq!(items, vec![1, 2, 3]);
    }

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
