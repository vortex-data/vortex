// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A runtime that segregates CPU-bound work from I/O work.
//!
//! This runtime uses tokio for async I/O and orchestration, and a dedicated Rayon
//! thread pool for CPU-bound work. This prevents CPU-heavy tasks from starving
//! network I/O operations.

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::OnceLock;

use futures::future::BoxFuture;
use rayon::ThreadPool;
use vortex_error::VortexExpect;
use vortex_error::vortex_panic;

use crate::runtime::AbortHandle;
use crate::runtime::AbortHandleRef;
use crate::runtime::Executor;
use crate::runtime::Handle;

/// Global CPU pool, lazily initialized.
/// This is shared across all CPUSegregatedRuntime handles.
static CPU_POOL: OnceLock<Arc<ThreadPool>> = OnceLock::new();

fn get_or_init_cpu_pool(reserved_for_io: usize) -> Arc<ThreadPool> {
    CPU_POOL
        .get_or_init(|| {
            let available = std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1);
            let cpu_threads = available.saturating_sub(reserved_for_io).max(1);

            Arc::new(
                rayon::ThreadPoolBuilder::new()
                    .num_threads(cpu_threads)
                    .thread_name(|i| format!("vortex-cpu-{}", i))
                    .build()
                    .unwrap_or_else(|e| vortex_panic!("Failed to create CPU thread pool: {}", e)),
            )
        })
        .clone()
}

/// A runtime that segregates CPU-bound work from I/O work.
///
/// - `spawn()` runs on the tokio runtime (for async orchestration and I/O)
/// - `spawn_cpu()` runs on a dedicated Rayon pool (bounded, leaves cores for I/O)
/// - `spawn_blocking()` runs on tokio's blocking pool (for blocking I/O)
///
/// This separation ensures that CPU-heavy work (like array decompression and
/// expression evaluation) doesn't starve network I/O operations, which need
/// timely attention to maintain TCP throughput.
pub struct CPUSegregatedRuntime;

impl CPUSegregatedRuntime {
    /// Create a [`Handle`] using the current tokio context, reserving 2 cores for I/O.
    ///
    /// The CPU pool is lazily initialized on first call and shared across all handles.
    pub fn current() -> Handle {
        Self::current_with_reserved(2)
    }

    /// Create a [`Handle`] using the current tokio context, reserving specified cores for I/O.
    ///
    /// Note: The `reserved_for_io` parameter only affects the first call that initializes
    /// the global CPU pool. Subsequent calls will reuse the existing pool.
    pub fn current_with_reserved(reserved_for_io: usize) -> Handle {
        // Initialize the CPU pool (or get existing one)
        drop(get_or_init_cpu_pool(reserved_for_io));

        // Use a static executor that grabs the current tokio handle on each call
        static EXECUTOR: LazyLock<Arc<dyn Executor>> =
            LazyLock::new(|| Arc::new(CurrentCPUSegregatedExecutor));

        Handle::new(Arc::downgrade(&EXECUTOR))
    }
}

/// An executor that uses the current tokio handle and the global CPU pool.
struct CurrentCPUSegregatedExecutor;

impl Executor for CurrentCPUSegregatedExecutor {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        Box::new(tokio::runtime::Handle::current().spawn(fut).abort_handle())
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        // Spawn on the dedicated CPU pool, not tokio.
        // This ensures CPU-heavy work doesn't block I/O threads.
        let cpu_pool = CPU_POOL
            .get()
            .vortex_expect("CPU pool not initialized - call CPUSegregatedRuntime::current() first");
        cpu_pool.spawn(move || {
            task();
        });
        // CPU tasks cannot be aborted once spawned to the Rayon pool
        Box::new(NoOpAbortHandle)
    }

    fn spawn_blocking(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        Box::new(
            tokio::runtime::Handle::current()
                .spawn_blocking(task)
                .abort_handle(),
        )
    }
}

/// A no-op abort handle for tasks that cannot be cancelled.
///
/// Rayon tasks cannot be aborted once they've been spawned to the pool.
struct NoOpAbortHandle;

impl AbortHandle for NoOpAbortHandle {
    fn abort(self: Box<Self>) {
        // Rayon tasks cannot be aborted once spawned
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use super::*;

    #[tokio::test]
    async fn test_cpu_segregated_spawn_cpu() {
        let handle = CPUSegregatedRuntime::current();

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let task = handle.spawn_cpu(move || {
            counter_clone.fetch_add(1, Ordering::SeqCst);
        });

        // Wait for completion
        task.await;
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn test_cpu_segregated_spawn() {
        let handle = CPUSegregatedRuntime::current();

        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let task = handle.spawn(async move {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            42
        });

        let result = task.await;
        assert_eq!(result, 42);
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }
}
