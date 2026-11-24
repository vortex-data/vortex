// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;
use std::time::Duration;

use parking_lot::Mutex;
use smol::block_on;
use vortex_error::VortexExpect;
use vortex_utils::parallelism::get_available_parallelism;

use crate::runtime::smol::SmolExecutor;

/// A handle to a background worker pool driving a
/// [`CurrentThreadRuntime`](crate::runtime::current::CurrentThreadRuntime).
///
/// Clones share the same underlying pool; workers shut down once the last handle is dropped.
#[derive(Clone)]
pub struct CurrentThreadWorkerPool {
    executor: Arc<SmolExecutor>,
    state: Arc<Mutex<PoolState>>,
}

impl CurrentThreadWorkerPool {
    pub(super) fn new(executor: Arc<SmolExecutor>) -> Self {
        Self {
            executor,
            state: Arc::new(Mutex::new(PoolState::default())),
        }
    }

    /// Set the number of worker threads to the available system parallelism as reported by
    /// [`get_available_parallelism()`] minus 1, to leave a slot open for the calling thread.
    pub fn set_workers_to_available_parallelism(&self) {
        let n = get_available_parallelism()
            .map(|n| n.saturating_sub(1).max(1))
            .unwrap_or(1);
        self.set_workers(n);
    }

    /// Set the number of worker threads
    /// - If n > current: spawns additional workers
    /// - If n < current: signals extra workers to shut down
    pub fn set_workers(&self, n: usize) {
        let mut state = self.state.lock();
        let current = state.workers.len();

        if n > current {
            // Spawn new workers
            for _ in current..n {
                let shutdown = Arc::new(AtomicBool::new(false));
                let executor = Arc::clone(&self.executor);
                let shutdown_clone = Arc::clone(&shutdown);

                std::thread::Builder::new()
                    .name("vortex-current-thread-worker".to_string())
                    .spawn(move || {
                        // Run the executor with a sleeping future that checks for shutdown
                        block_on(executor.run(async move {
                            while !shutdown_clone.load(Ordering::Relaxed) {
                                smol::Timer::after(Duration::from_millis(100)).await;
                            }
                        }))
                    })
                    .vortex_expect("Failed to spawn current thread worker");

                state.workers.push(WorkerHandle { shutdown });
            }
        } else if n < current {
            // Drop excess worker handles so their `Drop` impl signals shutdown.
            state.workers.truncate(n);
        }
    }

    /// Get the current number of worker threads
    pub fn worker_count(&self) -> usize {
        self.state.lock().workers.len()
    }
}

#[derive(Default)]
struct PoolState {
    /// The set of worker handles for the background threads.
    workers: Vec<WorkerHandle>,
}

struct WorkerHandle {
    /// The shutdown flag indicating that the worker should stop.
    shutdown: Arc<AtomicBool>,
}

impl Drop for WorkerHandle {
    fn drop(&mut self) {
        self.shutdown.store(true, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use std::sync::mpsc;
    use std::thread;
    use std::time::Instant;

    use super::*;
    use crate::runtime::BlockingRuntime;
    use crate::runtime::current::CurrentThreadRuntime;

    fn run_task(runtime: &CurrentThreadRuntime) {
        let (tx, rx) = mpsc::channel();
        runtime
            .handle()
            .spawn(async move {
                let _ = tx.send(());
            })
            .detach();
        rx.recv_timeout(Duration::from_secs(2))
            .expect("background worker should complete spawned task");
    }

    fn wait_for_worker_exit(shutdown: &Arc<AtomicBool>) {
        let deadline = Instant::now() + Duration::from_secs(2);
        while Arc::strong_count(shutdown) > 1 {
            assert!(
                Instant::now() < deadline,
                "worker thread did not exit after shutdown"
            );
            thread::sleep(Duration::from_millis(10));
        }
    }

    #[test]
    fn test_worker_pool_clone_and_shutdown() {
        let runtime = CurrentThreadRuntime::new();
        let pool = runtime.new_pool();
        pool.set_workers(1);
        let shutdown = Arc::clone(&pool.state.lock().workers[0].shutdown);

        run_task(&runtime);

        // Dropping an unused clone must leave the surviving owner's worker running.
        drop(pool.clone());
        assert_eq!(pool.worker_count(), 1);
        assert!(!shutdown.load(Ordering::Relaxed));
        run_task(&runtime);

        // Dropping the last owner must shut down workers.
        drop(pool);
        assert!(shutdown.load(Ordering::Relaxed));
        wait_for_worker_exit(&shutdown);
    }
}
