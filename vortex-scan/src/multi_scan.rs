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

type ArrayFuture = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;
type ScanBuilderFactory = Arc<SegQueue<Box<dyn FnOnce() -> ScanBuilder<ArrayRef> + Send + Sync>>>;

/// Coordinator to orchestrate multiple scan operations.
///
/// `MultiScan` allows to queue multiple scan operations in order to execute
/// them in parallel. In particular, this enables scanning multiple files.
#[derive(Default)]
pub struct MultiScan {
    scan_builder_factory: ScanBuilderFactory,
}

impl MultiScan {
    pub fn new() -> Self {
        Self::default()
    }

    /// `ScanBuilder`s are passed through closures to decouple how the they are created.
    pub fn with_scan_builders<I, F>(self, closures: I) -> Self
    where
        F: FnOnce() -> ScanBuilder<ArrayRef> + 'static + Send + Sync,
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
    pub fn new_scan_iterator(&self) -> MultiScanIterator {
        MultiScanIterator {
            scan_builder_factory: self.scan_builder_factory.clone(),
            local_pool: LocalPool::new(),
            polled_tasks: FuturesUnordered::new(),
            task_queue: SegQueue::new(),
        }
    }
}

/// Scan iterator to participate in a `MultiScan`.
pub struct MultiScanIterator {
    local_pool: LocalPool,
    polled_tasks: FuturesUnordered<ArrayFuture>,

    /// Thread-safe queue of closures that lazily produce [`ScanBuilder`] instances.
    /// This queue is shared across all iterators being created with `new_scan_iterator`.
    scan_builder_factory: ScanBuilderFactory,
    task_queue: SegQueue<ArrayFuture>,
}

impl MultiScanIterator {
    fn pop_scan_task(&self) -> Option<VortexResult<ArrayFuture>> {
        if let Some(array_future_tuple) = self.task_queue.pop() {
            return Some(Ok(array_future_tuple));
        }
        None
    }
}

impl Iterator for MultiScanIterator {
    type Item = VortexResult<ArrayRef>;

    fn next(&mut self) -> Option<VortexResult<ArrayRef>> {
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
                    Ok(future) => self.polled_tasks.push(future),
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
                Some(Ok(array)) => return Some(Ok(array)),
                Some(Err(e)) => return Some(Err(e)),
                None => continue, // Try next batch of futures
            }
        }
    }
}
