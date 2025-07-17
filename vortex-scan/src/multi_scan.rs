// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use crossbeam_queue::SegQueue;
use futures::executor::LocalPool;
use futures::future::BoxFuture;
use futures::stream::{FuturesUnordered, StreamExt};
use vortex_array::ArrayRef;
use vortex_error::VortexResult;

use crate::ScanBuilder;

type ArrayFuture<S> = BoxFuture<'static, VortexResult<Option<(ArrayRef, S)>>>;
type ScanBuilderFactory<S> =
    Arc<SegQueue<Box<(dyn FnOnce() -> ScanBuilder<(ArrayRef, S)> + Send + Sync)>>>;

/// Coordinator to orchestrate multiple scan operations.
///
/// `MultiScan` allows to queue multiple scan operations in order to execute
/// them in parallel. In particular, this enables scanning multiple files.
pub struct MultiScan<S> {
    scan_builder_factory: ScanBuilderFactory<S>,
}

impl<S> Default for MultiScan<S> {
    fn default() -> Self {
        Self::new()
    }
}

impl<S> MultiScan<S> {
    pub fn new() -> Self {
        Self {
            scan_builder_factory: Arc::new(SegQueue::new()),
        }
    }

    /// Add lazily constructed scan builders paired with their corresponding states.
    pub fn with_scan_builders<I, F>(self, closures: I) -> Self
    where
        F: FnOnce() -> ScanBuilder<(ArrayRef, S)> + 'static + Send + Sync,
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
    pub fn new_scan_iterator(&self) -> MultiScanIterator<S> {
        MultiScanIterator {
            scan_builder_factory: self.scan_builder_factory.clone(),
            local_pool: LocalPool::new(),
            polled_tasks: FuturesUnordered::new(),
            task_queue: SegQueue::new(),
        }
    }
}

/// Scan iterator to participate in a `MultiScan`.
pub struct MultiScanIterator<S> {
    local_pool: LocalPool,
    polled_tasks: FuturesUnordered<ArrayFuture<S>>,

    /// Thread-safe queue of closures that lazily produce [`ScanBuilder`] instances.
    /// This queue is shared across all iterators being created with `new_scan_iterator`.
    scan_builder_factory: ScanBuilderFactory<S>,
    task_queue: SegQueue<ArrayFuture<S>>,
}

impl<S> MultiScanIterator<S> {
    fn pop_scan_task(&self) -> Option<VortexResult<ArrayFuture<S>>> {
        if let Some(task_with_state) = self.task_queue.pop() {
            return Some(Ok(task_with_state));
        }
        None
    }
}

impl<S: Send + Sync + 'static> Iterator for MultiScanIterator<S> {
    type Item = VortexResult<(ArrayRef, S)>;

    fn next(&mut self) -> Option<VortexResult<(ArrayRef, S)>> {
        loop {
            // Queue up tasks if the thread local queue is almost empty.
            if self.task_queue.len() <= 4 {
                if let Some(scan_builder_fn) = self.scan_builder_factory.pop() {
                    let split_tasks = scan_builder_fn().build().ok()?.1;
                    for task in split_tasks {
                        self.task_queue.push(Box::pin(task));
                    }
                }
                // TODO(Alex): worksteal tasks from other threads
            }

            if let Some(work_result) = self.pop_scan_task() {
                match work_result {
                    Ok(task) => {
                        self.polled_tasks.push(task);
                    }
                    Err(e) => return Some(Err(e)),
                }
            }

            if self.task_queue.is_empty() && self.polled_tasks.is_empty() {
                // All tasks have been fully processed.
                return None;
            }

            let result = self.local_pool.run_until(async {
                while let Some(result) = self.polled_tasks.next().await {
                    match result {
                        Ok(Some(array)) => return Some(Ok(array)),
                        Ok(None) => continue,
                        Err(e) => return Some(Err(e)),
                    }
                }
                None
            });

            match result {
                Some(Ok(array)) => {
                    return Some(Ok(array));
                }
                Some(Err(e)) => return Some(Err(e)),
                None => continue, // Try next batch of futures
            }
        }
    }
}
