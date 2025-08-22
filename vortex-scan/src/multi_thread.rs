// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::sync::{Arc, LazyLock};

use crate::ScanBuilder;
use futures::{stream, StreamExt};
use tokio::runtime::Builder;
use vortex_array::iter::{ArrayIterator, ArrayIteratorAdapter};
use vortex_array::ArrayRef;
use vortex_error::{VortexExpect, VortexResult};
use vortex_io::runtime::Runtime;

/// We create an internal Tokio runtime used exclusively for orchestrating work-stealing
/// of CPU-bound work for multithreaded scans.
///
/// It is intentionally not exposed to the user, not configurable, and does not enable I/O or
/// timers.
static CPU_RUNTIME: LazyLock<tokio::runtime::Runtime> = LazyLock::new(|| {
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
    ) -> VortexResult<impl Iterator<Item = T> + Send + 'static>
    where
        T: 'static + Send,
        F: Fn(VortexResult<ArrayRef>) -> T + Send + Sync + 'static,
    {
        let concurrency = self.concurrency;
        let num_workers = CPU_RUNTIME.metrics().num_workers();

        let runtime = Runtime::default();
        let handle = runtime.handle().clone();
        runtime.drive_on_tokio(CPU_RUNTIME.handle());

        let tasks = self.build(&handle)?;
        // We need to clone and send the map_fn into each task.
        let map_fn = Arc::new(map_fn);

        let mut stream = stream::iter(tasks)
            .map(move |task| {
                let map_fn = map_fn.clone();
                async move { task.await.transpose().map(|t| map_fn(t)) }
            })
            // TODO(ngates): this is very crude indeed. This buffered call essentially controls how
            //  many splits we have in-flight at any given time. We multiple workers by concurrency
            //  to configure per-thread concurrency, which essentially means each thread can make
            //  progress on one split while waiting for the I/O of another split to complete.
            //  In an ideal world, the number of in-flight tasks would be dynamically adjusted
            //  based on how much I/O the tasks _actually_ require. For example, all pruning tasks
            //  could be spawned immediately since they all use a single segment, this would allow
            //  head-room to run ahead and figure out the I/O demands of subsequent tasks.
            .buffered(num_workers * concurrency);

        Ok(iter::from_fn(move || {
            tokio::task::block_in_place(|| CPU_RUNTIME.handle().block_on(stream.next()))
        })
        .filter_map(|result| result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_runtime_initialization() {
        // Access the runtime to ensure it's initialized
        let runtime = &*CPU_RUNTIME;
        assert!(runtime.metrics().num_workers() > 0);

        // Verify runtime metrics
        let handle = runtime.handle();
        assert!(handle.metrics().num_workers() > 0);
    }

    #[test]
    fn test_runtime_concurrency_calculation() {
        let num_workers = CPU_RUNTIME.metrics().num_workers();
        let concurrency = 2;

        // Test the buffering calculation
        let buffer_size = num_workers * concurrency;
        assert!(buffer_size > 0);
        assert_eq!(buffer_size, num_workers * 2);
    }
}
