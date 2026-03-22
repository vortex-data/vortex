// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A dedicated, optionally core-pinned thread pool for CPU-bound work.
//!
//! Unlike tokio's work-stealing scheduler, tasks submitted to this pool run on fixed threads that
//! do not migrate between cores. This preserves CPU cache locality for decode-heavy workloads
//! (bitunpacking, FoR, dictionary gather, etc.).
//!
//! ## Thread pinning
//!
//! Pinning is best-effort and platform-dependent:
//! - **Linux**: each worker is affinity-pinned to a specific core via `sched_setaffinity`.
//! - **Other platforms**: workers are dedicated threads but not pinned. The OS scheduler may
//!   still migrate them, though in practice a busy compute thread tends to stay put.
//!
//! Even without strict pinning, the pool provides isolation from tokio's work-stealing: a decode
//! task will never be interleaved with unrelated async I/O work on the same thread.
//!
//! ## Integration with DataFusion / Tokio
//!
//! When Vortex is embedded in a tokio-based engine (e.g., DataFusion), the caller cannot replace
//! the async runtime. [`PinnedExecutor`] wraps an existing [`Executor`] (typically tokio) and
//! overrides only [`spawn_cpu`][Executor::spawn_cpu], routing CPU work to the pinned pool while
//! async futures and blocking I/O remain on the original runtime.
//!
//! ```ignore
//! use std::sync::Arc;
//! use vortex_io::runtime::Handle;
//! use vortex_io::runtime::pinned_pool::{PinnedCpuPool, PinnedExecutor};
//! use vortex_io::runtime::tokio::TokioRuntime;
//!
//! let pool = PinnedCpuPool::with_available_parallelism();
//! let tokio_executor: Arc<dyn Executor> = /* from TokioRuntime */;
//! let pinned = Arc::new(PinnedExecutor::new(pool, tokio_executor));
//! let handle = Handle::new(Arc::downgrade(&(pinned as Arc<dyn Executor>)));
//! // handle.spawn_cpu(...) now runs on pinned threads
//! ```

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::thread::JoinHandle;

use futures::future::BoxFuture;
use tracing::debug;
use tracing::trace;
use tracing::warn;

use crate::runtime::AbortHandle;
use crate::runtime::AbortHandleRef;
use crate::runtime::Executor;

// ---------------------------------------------------------------------------
// Task representation
// ---------------------------------------------------------------------------

/// A boxed CPU task with an associated cancellation flag.
struct TaskEntry {
    task: Box<dyn FnOnce() + Send + 'static>,
    cancelled: Arc<AtomicBool>,
}

/// An [`AbortHandle`] that sets a cancellation flag checked before task execution.
///
/// Once a task has started running, it cannot be interrupted — cancellation is
/// cooperative and only takes effect if the task has not yet been picked up by a worker.
struct PinnedAbortHandle {
    cancelled: Arc<AtomicBool>,
}

impl AbortHandle for PinnedAbortHandle {
    fn abort(self: Box<Self>) {
        self.cancelled.store(true, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// PinnedCpuPool
// ---------------------------------------------------------------------------

/// A fixed-size thread pool where each worker is (best-effort) pinned to a CPU core.
///
/// Work is distributed via a shared MPMC channel — all workers pull from the same queue,
/// giving natural load balancing without explicit scheduling.
pub struct PinnedCpuPool {
    sender: Option<kanal::Sender<TaskEntry>>,
    workers: Option<Vec<JoinHandle<()>>>,
    num_workers: usize,
}

impl PinnedCpuPool {
    /// Creates a pool with `num_workers` threads, each pinned to core `i % num_cores`.
    ///
    /// If `num_workers` is 0, it is clamped to 1.
    pub fn new(num_workers: usize) -> Self {
        let num_workers = num_workers.max(1);
        let core_ids = available_core_ids();

        // Bounded channel: back-pressure if workers can't keep up. The bound is generous
        // enough that callers rarely block, while preventing unbounded memory growth under
        // sustained load.
        let (sender, receiver) = kanal::bounded(num_workers * 64);

        let mut workers = Vec::with_capacity(num_workers);
        for worker_idx in 0..num_workers {
            let rx = receiver.clone();
            let core_id = core_ids.as_ref().map(|ids| ids[worker_idx % ids.len()]);

            let handle = std::thread::Builder::new()
                .name(format!("vortex-cpu-{worker_idx}"))
                .spawn(move || {
                    // Best-effort pin to core.
                    if let Some(id) = core_id {
                        if pin_current_thread(id) {
                            trace!(worker = worker_idx, core = id, "pinned worker to core");
                        } else {
                            debug!(
                                worker = worker_idx,
                                core = id,
                                "failed to pin worker (continuing unpinned)"
                            );
                        }
                    }

                    worker_loop(worker_idx, rx);
                })
                .unwrap_or_else(|e| panic!("failed to spawn vortex-cpu-{worker_idx}: {e}"));

            workers.push(handle);
        }

        debug!(num_workers, "pinned CPU pool started");
        Self {
            sender: Some(sender),
            workers: Some(workers),
            num_workers,
        }
    }

    /// Creates a pool sized to [`std::thread::available_parallelism`], minus one thread
    /// to leave headroom for the async runtime driving I/O.
    pub fn with_available_parallelism() -> Self {
        let n = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1);
        Self::new(n)
    }

    /// Returns the number of worker threads in the pool.
    pub fn num_workers(&self) -> usize {
        self.num_workers
    }

    /// Submit a CPU-bound closure for execution on a pool worker.
    ///
    /// Returns an [`AbortHandleRef`] that can cancel the task if it hasn't started yet.
    pub fn submit(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        let cancelled = Arc::new(AtomicBool::new(false));
        let handle = Box::new(PinnedAbortHandle {
            cancelled: cancelled.clone(),
        });

        let entry = TaskEntry { task, cancelled };

        // If the pool has been shut down (sender taken), the task is silently dropped.
        match self.sender.as_ref() {
            Some(sender) => {
                if sender.send(entry).is_err() {
                    warn!("pinned CPU pool is shut down; task will not execute");
                }
            }
            None => {
                warn!("pinned CPU pool is shut down; task will not execute");
            }
        }

        handle
    }

    /// Gracefully shuts down the pool: drops the sender so workers drain remaining tasks
    /// and then exit. Blocks until all workers have joined.
    pub fn shutdown(mut self) {
        // Drop the sender to close the channel. Workers will drain remaining tasks
        // and then exit when they observe the closed channel.
        self.sender.take();

        // Join all workers to ensure in-flight CPU work completes before the caller
        // continues. This matters during process shutdown or test teardown.
        if let Some(workers) = self.workers.take() {
            for (i, handle) in workers.into_iter().enumerate() {
                if let Err(e) = handle.join() {
                    warn!(worker = i, "pinned worker panicked: {e:?}");
                }
            }
        }
        debug!("pinned CPU pool shut down");
    }
}

impl Drop for PinnedCpuPool {
    fn drop(&mut self) {
        // Drop the sender to signal workers to exit. We don't join here because `drop`
        // shouldn't block indefinitely. Workers will finish in-flight work and exit on
        // their own. For a clean shutdown, call `shutdown()` explicitly.

        // `self.sender` is dropped automatically, which closes the channel.
        // Workers are detached — their JoinHandles are dropped without joining.
    }
}

// ---------------------------------------------------------------------------
// Worker loop
// ---------------------------------------------------------------------------

fn worker_loop(worker_idx: usize, receiver: kanal::Receiver<TaskEntry>) {
    trace!(worker = worker_idx, "worker started");

    while let Ok(entry) = receiver.recv() {
        // Check cancellation before executing.
        if entry.cancelled.load(Ordering::Acquire) {
            trace!(worker = worker_idx, "skipping cancelled task");
            continue;
        }

        (entry.task)();
    }

    trace!(worker = worker_idx, "worker exiting (channel closed)");
}

// ---------------------------------------------------------------------------
// PinnedExecutor
// ---------------------------------------------------------------------------

/// An [`Executor`] that routes `spawn_cpu` to a [`PinnedCpuPool`] while delegating
/// `spawn` and `spawn_blocking_io` to a fallback executor (typically tokio).
///
/// This allows cache-local CPU decode work to coexist with an externally-provided
/// async runtime without replacing it.
pub struct PinnedExecutor {
    pool: Arc<PinnedCpuPool>,
    fallback: Arc<dyn Executor>,
}

impl PinnedExecutor {
    /// Creates a new `PinnedExecutor`.
    ///
    /// - `pool`: the dedicated CPU thread pool.
    /// - `fallback`: the async runtime for futures and blocking I/O (e.g., tokio).
    pub fn new(pool: Arc<PinnedCpuPool>, fallback: Arc<dyn Executor>) -> Self {
        Self { pool, fallback }
    }

    /// Returns a reference to the underlying pool.
    pub fn pool(&self) -> &PinnedCpuPool {
        &self.pool
    }
}

impl std::fmt::Debug for PinnedExecutor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("PinnedExecutor")
            .field("num_workers", &self.pool.num_workers)
            .finish()
    }
}

impl Executor for PinnedExecutor {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        // Async futures go to the fallback runtime (tokio).
        self.fallback.spawn(fut)
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        // CPU work goes to the pinned pool.
        self.pool.submit(task)
    }

    fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        // Blocking I/O goes to the fallback runtime (typically tokio::spawn_blocking).
        self.fallback.spawn_blocking_io(task)
    }
}

// ---------------------------------------------------------------------------
// Platform-specific thread pinning
// ---------------------------------------------------------------------------

/// Returns the list of available core IDs, or `None` if detection fails.
fn available_core_ids() -> Option<Vec<usize>> {
    #[cfg(target_os = "linux")]
    {
        linux::available_core_ids()
    }
    #[cfg(not(target_os = "linux"))]
    {
        // Fallback: use available_parallelism as a proxy for core count.
        let n = std::thread::available_parallelism().ok()?.get();
        Some((0..n).collect())
    }
}

/// Attempts to pin the current thread to the given core ID. Returns `true` on success.
///
/// Pinning is best-effort and platform-dependent:
/// - **Linux**: strict pinning via `sched_setaffinity`.
/// - **macOS**: uses `thread_policy_set` with `THREAD_AFFINITY_POLICY` (hint, not strict).
/// - **Other**: no-op, returns `false`.
///
/// Even without strict pinning, the dedicated pool avoids tokio work-stealing, which is
/// the primary source of cache-line bouncing.
fn pin_current_thread(core_id: usize) -> bool {
    #[cfg(target_os = "linux")]
    {
        linux::pin_current_thread(core_id)
    }
    #[cfg(target_os = "macos")]
    {
        macos::pin_current_thread(core_id)
    }
    #[cfg(not(any(target_os = "linux", target_os = "macos")))]
    {
        let _ = core_id;
        false
    }
}

#[cfg(target_os = "linux")]
mod linux {
    /// Returns the set of CPUs the current thread is allowed to run on.
    pub(super) fn available_core_ids() -> Option<Vec<usize>> {
        // SAFETY: cpu_set_t is POD. We zero-initialize then read via CPU_ISSET.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            let ret = libc::sched_getaffinity(0, size_of::<libc::cpu_set_t>(), &mut set);
            if ret != 0 {
                return None;
            }

            let mut cores = Vec::new();
            for i in 0..libc::CPU_SETSIZE as usize {
                if libc::CPU_ISSET(i, &set) {
                    cores.push(i);
                }
            }
            if cores.is_empty() { None } else { Some(cores) }
        }
    }

    /// Pin the calling thread to `core_id` using `sched_setaffinity`.
    pub(super) fn pin_current_thread(core_id: usize) -> bool {
        // SAFETY: cpu_set_t is POD. We set exactly one bit and call sched_setaffinity.
        unsafe {
            let mut set: libc::cpu_set_t = std::mem::zeroed();
            libc::CPU_SET(core_id, &mut set);
            libc::sched_setaffinity(0, size_of::<libc::cpu_set_t>(), &set) == 0
        }
    }
}

#[cfg(target_os = "macos")]
mod macos {
    /// Best-effort affinity on macOS using `thread_policy_set` with `THREAD_AFFINITY_POLICY`.
    ///
    /// macOS does not support strict core pinning. Affinity tags are scheduler hints that
    /// encourage (but do not guarantee) threads with the same tag to share a core's cache
    /// hierarchy. We assign each worker a unique tag derived from its core_id so the scheduler
    /// spreads them across cores.
    pub(super) fn pin_current_thread(core_id: usize) -> bool {
        // SAFETY: mach kernel calls with correct argument types and sizes.
        // thread_affinity_policy_data_t has a single i32 field: affinity_tag.
        // Tag 0 means "no affinity", so we offset by 1.
        unsafe {
            let mut tag: i32 = (core_id as i32) + 1;

            const THREAD_AFFINITY_POLICY: u32 = 4;
            const THREAD_AFFINITY_POLICY_COUNT: u32 = 1;

            unsafe extern "C" {
                fn mach_thread_self() -> u32;
                fn thread_policy_set(
                    thread: u32,
                    flavor: u32,
                    policy_info: *mut i32,
                    count: u32,
                ) -> i32;
            }

            let thread = mach_thread_self();
            let ret = thread_policy_set(
                thread,
                THREAD_AFFINITY_POLICY,
                &mut tag as *mut i32,
                THREAD_AFFINITY_POLICY_COUNT,
            );
            ret == 0 // KERN_SUCCESS
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;
    use std::time::Duration;

    use super::*;

    #[test]
    fn test_basic_spawn() {
        let pool = PinnedCpuPool::new(2);
        let counter = Arc::new(AtomicUsize::new(0));

        let n = 100;
        let mut handles = Vec::new();
        for _ in 0..n {
            let c = counter.clone();
            let (tx, rx) = std::sync::mpsc::channel();
            pool.submit(Box::new(move || {
                c.fetch_add(1, Ordering::SeqCst);
                let _ = tx.send(());
            }));
            handles.push(rx);
        }

        // Wait for all tasks to complete.
        for rx in handles {
            rx.recv_timeout(Duration::from_secs(5))
                .expect("task did not complete in time");
        }

        assert_eq!(counter.load(Ordering::SeqCst), n);
    }

    #[test]
    fn test_abort_before_execution() {
        // Use a pool with 1 worker and fill it with a blocking task so we can test cancellation.
        let pool = PinnedCpuPool::new(1);
        let counter = Arc::new(AtomicUsize::new(0));

        let (block_tx, block_rx) = std::sync::mpsc::channel::<()>();
        let (started_tx, started_rx) = std::sync::mpsc::channel::<()>();

        // Submit a blocking task to occupy the single worker.
        pool.submit(Box::new(move || {
            let _ = started_tx.send(());
            let _ = block_rx.recv(); // block until released
        }));

        // Wait for the blocker to start.
        started_rx
            .recv_timeout(Duration::from_secs(5))
            .expect("blocker did not start");

        // Now submit a task and immediately abort it.
        let c = counter.clone();
        let abort = pool.submit(Box::new(move || {
            c.fetch_add(1, Ordering::SeqCst);
        }));
        abort.abort();

        // Release the blocker so the worker can pick up the cancelled task.
        let _ = block_tx.send(());

        // Give the worker time to process.
        std::thread::sleep(Duration::from_millis(100));

        // The counter should remain 0 because the task was cancelled.
        assert_eq!(counter.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn test_shutdown_completes_inflight() {
        let pool = PinnedCpuPool::new(2);
        let counter = Arc::new(AtomicUsize::new(0));

        for _ in 0..50 {
            let c = counter.clone();
            pool.submit(Box::new(move || {
                c.fetch_add(1, Ordering::SeqCst);
            }));
        }

        // Shutdown should drain remaining tasks.
        pool.shutdown();
        assert_eq!(counter.load(Ordering::SeqCst), 50);
    }

    #[test]
    fn test_with_available_parallelism() {
        let pool = PinnedCpuPool::with_available_parallelism();
        assert!(pool.num_workers() >= 1);
        pool.shutdown();
    }

    #[test]
    fn test_tasks_run_on_different_threads() {
        let num_workers = 4;
        let pool = PinnedCpuPool::new(num_workers);
        let thread_ids: Arc<parking_lot::Mutex<Vec<std::thread::ThreadId>>> =
            Arc::new(parking_lot::Mutex::new(Vec::new()));

        // Submit exactly as many tasks as workers so the barrier doesn't deadlock.
        let mut rxs = Vec::new();
        let barrier = Arc::new(std::sync::Barrier::new(num_workers));

        for _ in 0..num_workers {
            let ids = thread_ids.clone();
            let b = barrier.clone();
            let (tx, rx) = std::sync::mpsc::channel();
            pool.submit(Box::new(move || {
                // Barrier ensures all tasks are running concurrently on different workers.
                b.wait();
                ids.lock().push(std::thread::current().id());
                let _ = tx.send(());
            }));
            rxs.push(rx);
        }

        for rx in rxs {
            rx.recv_timeout(Duration::from_secs(5))
                .expect("task did not complete");
        }

        let ids = thread_ids.lock();
        // With 4 workers and 4 concurrent tasks, we should see 4 distinct thread IDs.
        let unique: std::collections::HashSet<_> = ids.iter().collect();
        assert_eq!(
            unique.len(),
            num_workers,
            "expected each task to run on a different worker thread"
        );

        pool.shutdown();
    }
}
