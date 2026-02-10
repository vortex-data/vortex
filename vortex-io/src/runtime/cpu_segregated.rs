// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A runtime that segregates CPU-bound work from I/O work.
//!
//! This runtime uses tokio for async I/O and orchestration, and a dedicated Rayon
//! thread pool for CPU-bound work. This prevents CPU-heavy tasks from starving
//! network I/O operations.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use futures::future::BoxFuture;
use rayon::ThreadPool;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use crate::runtime::AbortHandle;
use crate::runtime::AbortHandleRef;
use crate::runtime::Executor;
use crate::runtime::Handle;

/// A runtime that segregates CPU-bound work from I/O work.
///
/// - `spawn()` runs on the tokio runtime (for async orchestration and I/O)
/// - `spawn_cpu()` runs on a dedicated Rayon pool (bounded, leaves cores for I/O)
/// - `spawn_blocking()` runs on tokio's blocking pool (for blocking I/O)
///
/// This separation ensures that CPU-heavy work (like array decompression and
/// expression evaluation) doesn't starve network I/O operations, which need
/// timely attention to maintain TCP throughput.
pub struct CPUSegregatedExecutorInner {
    cpu_pool: ThreadPool,
    io_pool: tokio::runtime::Runtime,
}

pub struct CPUSegregatedExecutor {
    owned: Arc<dyn Executor>,
}

impl CPUSegregatedExecutor {
    pub fn handle(&self) -> Handle {
        Handle::new(Arc::downgrade(&self.owned))
    }

    /// Create a [`Handle`] using the current tokio context, reserving 2 cores for I/O.
    ///
    /// The CPU pool is lazily initialized on first call and shared across all handles.
    pub fn current() -> VortexResult<Self> {
        Self::current_with_reserved(2)
    }

    /// Create a [`Handle`] using the current tokio context, reserving specified cores for I/O.
    ///
    /// Note: The `reserved_for_io` parameter only affects the first call that initializes
    /// the global CPU pool. Subsequent calls will reuse the existing pool.
    // TODO(DK): rename this to not use current
    pub fn current_with_reserved(reserved_for_io: usize) -> VortexResult<Self> {
        let available = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(1);
        let cpu_threads = available.saturating_sub(reserved_for_io).max(1);

        let cpu_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cpu_threads)
            .thread_name(|i| format!("vortex-cpu-{}", i))
            .build()
            .unwrap_or_else(|e| vortex_panic!("Failed to create CPU thread pool: {}", e));
        let io_pool = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(reserved_for_io)
            .thread_name_fn(|| {
                static ATOMIC_ID: AtomicUsize = AtomicUsize::new(0);
                let id = ATOMIC_ID.fetch_add(1, Ordering::SeqCst);
                format!("vortex-io-{}", id)
            })
            .enable_time()
            .enable_io()
            .build()?;

        let owned = Arc::from(CPUSegregatedExecutorInner { cpu_pool, io_pool });
        Ok(CPUSegregatedExecutor { owned })
    }
}

impl Executor for CPUSegregatedExecutorInner {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        Box::new(self.io_pool.spawn(fut).abort_handle())
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        // Spawn on the dedicated CPU pool, not tokio.
        // This ensures CPU-heavy work doesn't block I/O threads.
        self.cpu_pool.spawn(move || {
            task();
        });
        // CPU tasks cannot be aborted once spawned to the Rayon pool
        Box::new(NoOpAbortHandle)
    }

    fn spawn_blocking(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        Box::new(self.io_pool.spawn_blocking(task).abort_handle())
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
        let handle = CPUSegregatedExecutor::current();

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
        let handle = CPUSegregatedExecutor::current();

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
