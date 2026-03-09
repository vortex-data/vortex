// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use crate::runtime::{AbortHandle, AbortHandleRef, BlockingRuntime, Executor, Handle};
use futures::future::BoxFuture;
use rayon::ThreadPool;
use tokio::runtime::Builder;
use vortex_error::vortex_panic;

use crate::runtime::available_cores::{AvailableCores, available_cores};

/// A runtime that segregates CPU-bound work from I/O work.
///
/// - `spawn()` runs on the Tokio runtime (for async orchestration and I/O)
/// - `spawn_cpu()` runs on a dedicated Rayon pool (bounded, leaves cores for I/O)
/// - `spawn_blocking_io()` runs on Tokio's blocking pool (for blocking I/O)
///
/// This separation ensures that CPU-heavy work (like array decompression and
/// expression evaluation) doesn't starve network I/O operations, which need
/// timely attention to maintain TCP throughput.
pub struct CpuSegregatedExecutor {
    cpu_thread_count: usize,
    io_thread_count: usize,
    cpu_pool: ThreadPool,
    io_pool: tokio::runtime::Runtime,
}

pub struct CpuSegregatedRuntime {
    owned: Arc<CpuSegregatedExecutor>,
}

impl Default for CpuSegregatedRuntime {
    /// See [crate::available_cores::available_cores] for an explanation of logical and physical
    /// cores and why CPU-bound work achieves peak throughput when CPU threads correspond 1:1 with
    /// physical cores whereas I/O-bound work can achieve higher throughput when I/O threads
    /// correspond 1:1 with logical cores.
    fn default() -> Self {
        let AvailableCores { logical, physical } = available_cores();
        CpuSegregatedRuntime::new(physical.max(1), logical.max(1))
    }
}

impl CpuSegregatedRuntime {
    pub fn new(cpu_thread_count: usize, io_thread_count: usize) -> Self {
        assert!(cpu_thread_count > 0);
        assert!(io_thread_count > 0);

        let cpu_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(cpu_thread_count)
            .thread_name(|i| format!("vortex-cpu-{}", i))
            .build()
            .unwrap_or_else(|e| vortex_panic!("Failed to create CPU thread pool: {}", e));
        let io_pool = Builder::new_multi_thread()
            .worker_threads(io_thread_count)
            .thread_name_fn(|| {
                static ATOMIC_ID: AtomicUsize = AtomicUsize::new(0);
                let id = ATOMIC_ID.fetch_add(1, Ordering::SeqCst);
                format!("vortex-io-{}", id)
            })
            .enable_time()
            .enable_io()
            .build()
            .expect("Tokio did not start successfully.");

        let owned = Arc::from(CpuSegregatedExecutor {
            cpu_thread_count,
            io_thread_count,
            cpu_pool,
            io_pool,
        });
        CpuSegregatedRuntime { owned }
    }

    pub fn cpu_thread_count(&self) -> usize {
        self.owned.cpu_thread_count
    }

    pub fn io_thread_count(&self) -> usize {
        self.owned.io_thread_count
    }
}

impl Executor for CpuSegregatedExecutor {
    fn spawn(&self, fut: BoxFuture<'static, ()>) -> AbortHandleRef {
        Box::new(self.io_pool.spawn(fut).abort_handle())
    }

    fn spawn_cpu(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        let stop = Arc::new(AtomicBool::new(false));
        let should_stop = stop.clone();
        self.cpu_pool.spawn(move || {
            if should_stop.load(Ordering::Acquire) {
                return;
            }
            task();
        });
        Box::new(RayonAbortHandle(stop))
    }

    fn spawn_blocking_io(&self, task: Box<dyn FnOnce() + Send + 'static>) -> AbortHandleRef {
        Box::new(self.io_pool.spawn_blocking(task).abort_handle())
    }
}

// TODO(DK): I am confused by "BlockingRuntime" and "Executor" using different nouns. Maybe this
// should be BlockingExecutor?
impl BlockingRuntime for CpuSegregatedRuntime {
    type BlockingIterator<'a, R: 'a> = BlockingIterator<'a, R>;

    fn handle(&self) -> Handle {
        let temp = Arc::downgrade(&self.owned);
        Handle::new(temp)
    }

    fn block_on<Fut, R>(&self, fut: Fut) -> R
    where
        Fut: Future<Output = R>,
    {
        let handle = self.owned.io_pool.handle().clone();
        tokio::task::block_in_place(move || handle.block_on(fut))
    }

    fn block_on_stream<'a, S, R>(&self, stream: S) -> Self::BlockingIterator<'a, R>
    where
        S: futures::Stream<Item = R> + Send + 'a,
        R: Send + 'a,
    {
        let handle = self.owned.io_pool.handle().clone();
        let stream = Box::pin(stream);
        BlockingIterator { handle, stream }
    }
}

pub struct BlockingIterator<'a, T> {
    handle: tokio::runtime::Handle,
    stream: futures::stream::BoxStream<'a, T>,
}

impl<T> Iterator for BlockingIterator<'_, T> {
    type Item = T;

    fn next(&mut self) -> Option<Self::Item> {
        use futures::StreamExt;

        tokio::task::block_in_place(|| self.handle.block_on(self.stream.next()))
    }
}

struct RayonAbortHandle(Arc<AtomicBool>);

impl AbortHandle for RayonAbortHandle {
    fn abort(self: Box<Self>) {
        self.0.store(true, Ordering::Release);
    }
}
