//! Driver layer.
//!
//! A driver decides *who* runs each `(operator, lane)` pair: which
//! thread, on top of which async substrate, with what cross-shard
//! plumbing. The scheduler turn loop is unchanged; the driver just
//! decides where it executes.
//!
//! `CurrentThreadDriver` is the V1 reference driver, modelled on
//! `vortex_io::runtime::CurrentThreadRuntime`. It holds an
//! `Arc<smol::Executor<'static>>` and does no work on its own —
//! forward progress happens only when something calls `block_on` or
//! a worker pool is set to drive the executor in the background.
//!
//! The driver is `Clone`. Cloning hands out another handle to the
//! same executor; cloned handles can be used from multiple threads
//! simultaneously, each driving the shared executor.
//!
//! Cross-driver work today: the spawned graph runs synchronously
//! inside the spawned future (no awaits between scheduler turns).
//! With N pool workers and M spawned graphs, up to `min(N, M)`
//! graphs make progress in parallel. Cooperative yielding between
//! turns is a follow-up that requires factoring the scheduler turn
//! loop behind a `Worker` trait — see `docs/design/drivers.md`.

use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;

use parking_lot::Mutex;

use crate::FakeDriverIo;
use crate::EngineError;
use crate::EngineResult;
use crate::ExecutionMetrics;
use crate::OperatorGraph;
use crate::PreparedTask;
use crate::TaskOptions;
use crate::TaskReport;
use crate::TurnOutcome;

pub type DriverTask<T> = smol::Task<T>;

#[derive(Clone, Default)]
pub struct CurrentThreadDriver {
    executor: Arc<smol::Executor<'static>>,
}

impl CurrentThreadDriver {
    pub fn new() -> Self {
        Self::default()
    }

    /// Drive a future to completion on the calling thread. Also
    /// services any other tasks spawned on this driver.
    pub fn block_on<F: Future>(&self, fut: F) -> F::Output {
        smol::block_on(self.executor.run(fut))
    }

    /// Spawn a graph for execution. Returns a task whose output
    /// resolves to the graph's `TaskReport` and the metrics handle
    /// the run accumulated into.
    ///
    /// The future drives the graph cooperatively: one scheduler turn,
    /// then `yield_now().await` so other tasks on the executor get a
    /// chance to run, repeat. With one executor thread and N spawned
    /// graphs, the graphs interleave at turn granularity instead of
    /// hogging the worker for their whole duration.
    pub fn spawn(
        &self,
        graph: OperatorGraph,
        options: TaskOptions,
    ) -> DriverTask<EngineResult<SpawnedReport>> {
        let metrics = Arc::new(Mutex::new(ExecutionMetrics::default()));
        self.spawn_with_metrics(graph, metrics, options)
    }

    /// Variant of `spawn` that lets the caller provide their own
    /// metrics handle. Useful when the caller wants to inspect
    /// in-progress metrics from outside the spawned task.
    pub fn spawn_with_metrics(
        &self,
        graph: OperatorGraph,
        metrics: Arc<Mutex<ExecutionMetrics>>,
        options: TaskOptions,
    ) -> DriverTask<EngineResult<SpawnedReport>> {
        let metrics_handle = Arc::clone(&metrics);
        self.executor.spawn(async move {
            let max_turns = options.max_turns;
            let mut task = PreparedTask::prepare(graph, metrics, options)?;
            // The driver owns the I/O substrate. For V1 it's the
            // prototype `FakeDriverIo`; future driver-family
            // substrates (`BlockingFileIo`, `IoUringFileIo`, ...)
            // slot in here. The substrate is passed into every
            // engine entry point — `PreparedTask` itself never
            // owns it.
            let mut io = FakeDriverIo::new();
            // Multi-worker fast path: persistent OS thread pool
            // wraps the entire task. Workers are spawned once at
            // task start and signaled per turn via crossbeam_channel.
            // No yielding to the smol executor between turns —
            // multi-worker tasks dominate their runtime budget; if
            // multi-graph fairness becomes important we'll revisit.
            if task.worker_count() > 1 {
                match run_task_with_pool(&mut task, &mut io, max_turns)? {
                    TurnOutcome::Done => {
                        return Ok(SpawnedReport {
                            report: task.into_report(),
                            metrics: metrics_handle,
                        });
                    }
                    other => {
                        return Err(EngineError::message(format!(
                            "pool returned non-Done outcome: {other:?}"
                        )));
                    }
                }
            }
            // Single-worker path: synchronous turn loop with
            // cooperative yielding so multiple graphs spawned on the
            // same executor interleave at turn granularity.
            for _ in 0..max_turns {
                let outcome = task.turn(&mut io)?;
                match outcome {
                    TurnOutcome::Made | TurnOutcome::Idle => {
                        smol::future::yield_now().await;
                    }
                    TurnOutcome::Done => {
                        return Ok(SpawnedReport {
                            report: task.into_report(),
                            metrics: metrics_handle,
                        });
                    }
                }
            }
            Err(EngineError::message(
                "operator graph exceeded scheduler turn limit",
            ))
        })
    }

    /// Create a worker pool that drives this driver's executor in
    /// background threads. The pool starts with no workers; call
    /// `set_workers(n)` to spawn or shrink the pool.
    pub fn new_pool(&self) -> CurrentThreadPool {
        CurrentThreadPool::new(Arc::clone(&self.executor))
    }

    /// Get the underlying executor. Useful when callers want to
    /// spawn arbitrary `Send + 'static` futures on the same driver.
    pub fn executor(&self) -> Arc<smol::Executor<'static>> {
        Arc::clone(&self.executor)
    }
}

/// A spawned graph's outcome plus the metrics handle the run
/// accumulated into.
pub struct SpawnedReport {
    pub report: TaskReport,
    pub metrics: Arc<Mutex<ExecutionMetrics>>,
}


/// Persistent OS thread pool that drives multi-worker
/// [`PreparedTask`]s. Spawns N OS threads at construction; each
/// thread idles on a per-worker `crossbeam_channel` and runs one
/// `worker_turn` per signal. Reusing the same pool across many
/// tasks (e.g. benchmark iterations, planner re-plans) avoids the
/// thread-creation overhead of `std::thread::scope`.
///
/// ## Safety
///
/// Each `run_task` call passes a `*const PreparedTask` (encoded as
/// `usize`) to workers via the per-turn signal. The pointer is
/// valid only during a single `run_task` call; workers must not
/// retain it across calls. This is enforced by the protocol:
///
/// - The `TurnTick` value is consumed inside the worker loop and
///   not stored anywhere.
/// - Main's `&mut PreparedTask` is held only during `turn_phase1`,
///   `turn_phase2_admit_brokers`, `propagate_requirements`,
///   `rebalance_memory`, and `turn_phase3_classify`. Workers deref
///   the task only between `work_rx.recv()` (signalled by main
///   from outside those phases) and `done_tx.send()` (acked back to
///   main before main re-takes the `&mut`).
/// - These windows alternate strictly; no `&mut`/`&` aliasing.
///
/// On `Drop`, the pool drops its work senders, which unblocks
/// workers' `recv()` with `Disconnected`; each worker exits cleanly
/// and the pool joins them.
pub struct EngineWorkerPool {
    work_txs: Vec<crossbeam_channel::Sender<TurnTick>>,
    done_rx: crossbeam_channel::Receiver<EngineResult<bool>>,
    threads: Vec<std::thread::JoinHandle<()>>,
    n: usize,
}

#[derive(Clone, Copy)]
struct TurnTick {
    /// `*const PreparedTask` cast to `usize`. Sent fresh per turn so
    /// the pool can drive different `PreparedTask`s across calls.
    task_addr: usize,
    worker_id: usize,
}

impl EngineWorkerPool {
    /// Spawn `n` worker threads. `n` must be ≥ 1.
    pub fn new(n: usize) -> Self {
        use crate::WorkerId;
        use crossbeam_channel::bounded;

        assert!(n >= 1, "EngineWorkerPool requires at least one worker");
        let (work_txs, work_rxs): (Vec<_>, Vec<_>) =
            (0..n).map(|_| bounded::<TurnTick>(1)).unzip();
        let (done_tx, done_rx) = bounded::<EngineResult<bool>>(n);
        let mut threads = Vec::with_capacity(n);
        for (i, work_rx) in work_rxs.into_iter().enumerate() {
            let done_tx = done_tx.clone();
            let handle = std::thread::Builder::new()
                .name(format!("vortex-engine-worker-{i}"))
                .spawn(move || {
                    while let Ok(tick) = work_rx.recv() {
                        // SAFETY: see safety notes on `EngineWorkerPool`.
                        // `tick.task_addr` is valid for the duration of
                        // the active `run_task` call; the worker only
                        // dereferences it between `recv` and `send`.
                        let task_ref = unsafe {
                            &*(tick.task_addr as *const PreparedTask)
                        };
                        let ctx = task_ref.worker_ctx();
                        let result =
                            PreparedTask::worker_turn(ctx, WorkerId(tick.worker_id));
                        if done_tx.send(result).is_err() {
                            // Pool is being dropped; receiver is gone.
                            break;
                        }
                    }
                })
                .expect("spawn engine worker");
            threads.push(handle);
        }
        // We retain `done_tx` only inside this constructor; once
        // dropped the receiver remains live (held by `done_rx`).
        // Workers each clone their own `done_tx`; those clones live
        // until each thread exits.
        drop(done_tx);
        Self {
            work_txs,
            done_rx,
            threads,
            n,
        }
    }

    pub fn worker_count(&self) -> usize {
        self.n
    }

    /// Drive `task` to `TurnOutcome::Done`, or to the `max_turns`
    /// limit. The pool's worker count must match `task.worker_count()`.
    pub fn run_task(
        &self,
        task: &mut PreparedTask,
        io: &mut dyn crate::DriverIo,
        max_turns: usize,
    ) -> EngineResult<TurnOutcome> {
        use crate::TurnPhase1;

        assert_eq!(
            task.worker_count(),
            self.n,
            "pool worker count ({}) does not match task ({})",
            self.n,
            task.worker_count(),
        );
        let task_addr = task as *const PreparedTask as usize;

        for _turn in 0..max_turns {
            let phase1 = task.turn_phase1(io)?;
            let mut accumulated_phase1 = TurnPhase1 {
                async_wake: phase1.async_wake,
                substrate_activity: phase1.substrate_activity,
                propagated: phase1.propagated,
                grants_changed: phase1.grants_changed,
            };
            let mut any_progress = task.turn_phase2_admit_brokers(io)?;

            loop {
                for (i, tx) in self.work_txs.iter().enumerate() {
                    tx.send(TurnTick {
                        task_addr,
                        worker_id: i,
                    })
                    .expect("engine worker exited unexpectedly");
                }
                let mut any_made = false;
                for _ in 0..self.n {
                    if self
                        .done_rx
                        .recv()
                        .expect("engine worker exited unexpectedly")?
                    {
                        any_made = true;
                    }
                }
                if any_made {
                    any_progress = true;
                    task.propagate_requirements()?;
                    if task.rebalance_memory() {
                        task.mark_all_dirty(crate::DirtyCause::ExternalWake);
                    }
                } else {
                    break;
                }
            }

            let outcome =
                task.turn_phase3_classify(any_progress, accumulated_phase1, io)?;
            accumulated_phase1 = TurnPhase1::default();
            let _ = accumulated_phase1;
            match outcome {
                TurnOutcome::Done => return Ok(TurnOutcome::Done),
                TurnOutcome::Made | TurnOutcome::Idle => continue,
            }
        }
        Err(EngineError::message(
            "operator graph exceeded scheduler turn limit",
        ))
    }

    /// Convenience: prepare and run `graph` to completion on this
    /// pool. Returns the engine result; the report is discarded.
    pub fn run_graph(
        &self,
        graph: OperatorGraph,
        metrics: Arc<Mutex<ExecutionMetrics>>,
        options: TaskOptions,
    ) -> EngineResult<()> {
        let max_turns = options.max_turns;
        let mut task = PreparedTask::prepare(graph, metrics, options)?;
        let mut io = FakeDriverIo::new();
        match self.run_task(&mut task, &mut io, max_turns)? {
            TurnOutcome::Done => Ok(()),
            other => Err(EngineError::message(format!(
                "pool returned non-Done outcome: {other:?}"
            ))),
        }
    }
}

impl Drop for EngineWorkerPool {
    fn drop(&mut self) {
        // Closing the work senders unblocks workers' `recv` with
        // `Disconnected`, prompting clean exit.
        self.work_txs.clear();
        for t in self.threads.drain(..) {
            drop(t.join());
        }
    }
}

/// One-shot multi-worker drive used by `CurrentThreadDriver::spawn_with_metrics`.
///
/// Builds a fresh [`EngineWorkerPool`] sized to `task.worker_count()`,
/// runs the task to completion, and drops the pool. Callers that
/// need to reuse the pool across many tasks should construct an
/// `EngineWorkerPool` directly and call [`EngineWorkerPool::run_task`].
fn run_task_with_pool(
    task: &mut PreparedTask,
    io: &mut dyn crate::DriverIo,
    max_turns: usize,
) -> EngineResult<TurnOutcome> {
    let n = task.worker_count();
    debug_assert!(n > 1, "single-worker tasks should use the synchronous path");
    let pool = EngineWorkerPool::new(n);
    pool.run_task(task, io, max_turns)
}

/// Pool of background worker threads driving a `CurrentThreadDriver`'s
/// executor. Mirrors `vortex_io::runtime::CurrentThreadWorkerPool`.
#[derive(Clone)]
pub struct CurrentThreadPool {
    executor: Arc<smol::Executor<'static>>,
    state: Arc<Mutex<PoolState>>,
}

impl CurrentThreadPool {
    fn new(executor: Arc<smol::Executor<'static>>) -> Self {
        Self {
            executor,
            state: Arc::new(Mutex::new(PoolState::default())),
        }
    }

    /// Set the number of background worker threads.
    ///
    /// - `n` greater than the current count spawns extra workers;
    /// - `n` less signals workers to shut down (each worker checks
    ///   the flag periodically and exits when it sees true).
    ///
    /// Worker threads are detached; on shutdown they clean up after
    /// their own future returns and the OS reclaims the thread.
    pub fn set_workers(&self, n: usize) {
        let mut state = self.state.lock();
        let current = state.workers.len();
        if n > current {
            for _ in current..n {
                let shutdown = Arc::new(AtomicBool::new(false));
                let executor = Arc::clone(&self.executor);
                let shutdown_clone = Arc::clone(&shutdown);
                std::thread::Builder::new()
                    .name("vortex-engine-current-thread-worker".into())
                    .spawn(move || {
                        smol::block_on(executor.run(async move {
                            while !shutdown_clone.load(Ordering::Relaxed) {
                                smol::Timer::after(Duration::from_millis(50)).await;
                            }
                        }))
                    })
                    .expect("spawn current-thread worker");
                state.workers.push(WorkerHandle { shutdown });
            }
        } else if n < current {
            while state.workers.len() > n {
                if let Some(worker) = state.workers.pop() {
                    worker.shutdown.store(true, Ordering::Relaxed);
                }
            }
        }
    }

    pub fn worker_count(&self) -> usize {
        self.state.lock().workers.len()
    }
}

#[derive(Default)]
struct PoolState {
    workers: Vec<WorkerHandle>,
}

struct WorkerHandle {
    shutdown: Arc<AtomicBool>,
}

impl Drop for CurrentThreadPool {
    fn drop(&mut self) {
        let mut state = self.state.lock();
        for worker in state.workers.drain(..) {
            worker.shutdown.store(true, Ordering::Relaxed);
        }
    }
}
