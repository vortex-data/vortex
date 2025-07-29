// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::sync::{Arc, LazyLock};

use futures::future::BoxFuture;
use tokio::runtime::{Builder, Runtime};
use vortex_array::ArrayRef;
use vortex_array::iter::{ArrayIterator, ArrayIteratorAdapter};
use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::ScanBuilder;

/// We create an internal Tokio runtime used exclusively for orchestrating work-stealing
/// of CPU-bound work for multithreaded scans.
///
/// It is intentionally not exposed to the user, not configurable, and does not enable I/O or
/// timers.
static CPU_RUNTIME: LazyLock<Runtime> = LazyLock::new(|| {
    Builder::new_multi_thread()
        .thread_name("vortex-multithread-scan")
        .build()
        .vortex_expect("Failed to create a new Tokio runtime")
});

impl ScanBuilder<ArrayRef> {
    /// Execute the scan on multiple worker threads.
    pub fn into_array_iter_multithread(self) -> VortexResult<impl ArrayIterator + Send + 'static> {
        let dtype = self.dtype()?;
        Ok(ArrayIteratorAdapter::new(
            dtype,
            self.into_iter_multithread(|a| a)?,
        ))
    }

    /// Execute the scan on multiple worker threads.
    ///
    /// A `map_fn` can be passed to further transform the results of the scan while still running
    /// on the thread pool.
    pub fn into_iter_multithread<T, F>(
        self,
        map_fn: F,
    ) -> VortexResult<Box<dyn Iterator<Item = T> + Send>>
    where
        T: 'static + Send,
        F: Fn(VortexResult<ArrayRef>) -> T + Send + Sync + 'static,
    {
        let concurrency = self.concurrency;
        let num_workers = CPU_RUNTIME.metrics().num_workers();
        let max_concurrent = num_workers * concurrency;

        let tasks = self.build()?;
        let map_fn = Arc::new(map_fn);
        let handle = CPU_RUNTIME.handle().clone();

        // State for the iterator
        struct IterState<T, F> {
            remaining_tasks: std::vec::IntoIter<BoxFuture<'static, VortexResult<Option<ArrayRef>>>>,
            active_handles: Vec<tokio::task::JoinHandle<Option<T>>>,
            max_concurrent: usize,
            handle: tokio::runtime::Handle,
            map_fn: Arc<F>,
        }

        let mut state = IterState {
            remaining_tasks: tasks.into_iter(),
            active_handles: Vec::new(),
            max_concurrent,
            handle,
            map_fn,
        };

        // Fill initial pool
        while state.active_handles.len() < state.max_concurrent {
            if let Some(task) = state.remaining_tasks.next() {
                let map_fn = state.map_fn.clone();
                let join_handle = state
                    .handle
                    .spawn(async move { task.await.transpose().map(|t| map_fn(t)) });
                state.active_handles.push(join_handle);
            } else {
                break;
            }
        }

        Ok(Box::new(
            iter::from_fn(move || {
                if state.active_handles.is_empty() {
                    return None;
                }

                let join_handle = state.active_handles.remove(0);

                if let Some(task) = state.remaining_tasks.next() {
                    let map_fn = state.map_fn.clone();
                    let new_handle = state
                        .handle
                        .spawn(async move { task.await.transpose().map(|t| map_fn(t)) });
                    state.active_handles.push(new_handle);
                }

                let result = if tokio::runtime::Handle::try_current().is_ok() {
                    tokio::task::block_in_place(|| CPU_RUNTIME.handle().block_on(join_handle))
                } else {
                    futures::executor::block_on(join_handle)
                };

                Some(result)
            })
            .filter_map(|result| {
                result
                    .map_err(|e| vortex_err!("Failed to join on a spawned scan task {e}"))
                    .vortex_expect("Failed to join on a spawned scan task")
            }),
        ))
    }
}
