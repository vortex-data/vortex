// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use crossbeam_queue::SegQueue;
use futures::executor::LocalPool;
use futures::future::BoxFuture;
use vortex_error::VortexResult;

use crate::ScanBuilder;

type ArrayFuture<T> = BoxFuture<'static, VortexResult<Option<T>>>;
type ScanBuilderFactory<T> = Arc<SegQueue<Box<(dyn FnOnce() -> ScanBuilder<T> + Send + Sync)>>>;

/// Coordinator to orchestrate multiple scan operations.
///
/// `MultiScan` allows to queue multiple scan operations in order to execute
/// them in parallel. In particular, this enables scanning multiple files.
#[derive(Default)]
pub struct MultiScan<T> {
    scan_builder_factory: ScanBuilderFactory<T>,
}

impl<T> MultiScan<T> {
    pub fn new() -> Self {
        Self {
            scan_builder_factory: Arc::new(SegQueue::new()),
        }
    }

    /// Add lazily constructed scan builders paired with their corresponding states.
    pub fn with_scan_builders<I, F>(self, closures: I) -> Self
    where
        F: FnOnce() -> ScanBuilder<T> + 'static + Send + Sync,
        I: IntoIterator<Item = F>,
    {
        for closure in closures.into_iter() {
            self.scan_builder_factory.push(Box::new(closure));
        }

        self
    }

    /// Creates a new iterator to participate in the scan.
    ///
    /// The scan progresses when calling `next` on the iterator.
    pub fn new_scan_iterator(&self) -> MultiScanIterator<T> {
        MultiScanIterator {
            scan_builder_factory: self.scan_builder_factory.clone(),
            local_pool: LocalPool::new(),
            task_queue: SegQueue::new(),
        }
    }
}

/// Scan iterator to participate in a `MultiScan`.
pub struct MultiScanIterator<T> {
    local_pool: LocalPool,

    /// Thread-safe queue of closures that lazily produce [`ScanBuilder`] instances.
    /// This queue is shared across all iterators being created with `new_scan_iterator`.
    scan_builder_factory: ScanBuilderFactory<T>,
    task_queue: SegQueue<ArrayFuture<T>>,
}

impl<T: Send + Sync + 'static> Iterator for MultiScanIterator<T> {
    type Item = VortexResult<T>;

    fn next(&mut self) -> Option<VortexResult<T>> {
        // Queue up tasks if the thread local queue is empty.
        if self.task_queue.is_empty() {
            if let Some(scan_builder_fn) = self.scan_builder_factory.pop() {
                match scan_builder_fn().build() {
                    Ok(tasks) => {
                        for task in tasks.1 {
                            self.task_queue.push(Box::pin(task));
                        }
                    }
                    Err(err) => return Some(Err(err)),
                }
            }
            // TODO(Alex): worksteal tasks from other threads
        }

        let task = self.task_queue.pop()?;

        match self.local_pool.run_until(async { task.await }) {
            Ok(task) => return Some(Ok(task?)),
            Err(err) => return Some(Err(err)),
        }
    }
}
