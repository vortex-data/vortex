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

#[derive(Clone)]
pub struct CurrentThreadWorkerPool {
    executor: Arc<smol::Executor<'static>>,
    state: Arc<Mutex<PoolState>>,
}

impl CurrentThreadWorkerPool {
    pub(super) fn new(executor: Arc<smol::Executor<'static>>) -> Self {
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
            // Signal extra workers to shutdown and remove them
            while state.workers.len() > n {
                if let Some(worker) = state.workers.pop() {
                    worker.shutdown.store(true, Ordering::Relaxed);
                }
            }
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

impl Drop for CurrentThreadWorkerPool {
    fn drop(&mut self) {
        let mut state = self.state.lock();

        // Signal all workers to shut down
        for worker in state.workers.drain(..) {
            worker.shutdown.store(true, Ordering::Relaxed);
        }
    }
}
