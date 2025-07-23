// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::executor::LocalPool;
use futures::future::BoxFuture;
use vortex_error::VortexResult;

use crate::work_queue::{TaskFactory, WorkQueue, WorkQueueIterator};

pub type ArrayFuture<T> = BoxFuture<'static, VortexResult<Option<T>>>;

/// Coordinator to orchestrate multiple scan operations.
///
/// `MultiScan` allows to queue multiple scan operations in order to execute
/// them in parallel. In particular, this enables scanning multiple files.
pub struct MultiScan<T> {
    work_queue: WorkQueue<ArrayFuture<T>>,
}

impl<T: 'static + Send + Sync> MultiScan<T> {
    /// Created with lazily constructed scan builders closures.
    pub fn new<I, F>(closures: I) -> Self
    where
        F: FnOnce() -> VortexResult<Vec<ArrayFuture<T>>> + 'static + Send + Sync,
        I: IntoIterator<Item = F>,
    {
        Self {
            work_queue: WorkQueue::new(
                closures
                    .into_iter()
                    .map(|closure| Box::new(closure) as TaskFactory<ArrayFuture<T>>),
            ),
        }
    }

    /// Creates a new iterator to participate in the scan.
    ///
    /// The scan progresses when calling `next` on the iterator.
    pub fn new_scan_iterator(&self) -> MultiScanIterator<T> {
        MultiScanIterator {
            inner: self.work_queue.new_iterator(),
            local_pool: Default::default(),
        }
    }
}

/// Scan iterator to participate in a `MultiScan`.
pub struct MultiScanIterator<T> {
    inner: WorkQueueIterator<ArrayFuture<T>>,
    local_pool: LocalPool,
}

impl<T: Send + Sync + 'static> Iterator for MultiScanIterator<T> {
    type Item = VortexResult<T>;

    fn next(&mut self) -> Option<VortexResult<T>> {
        match self.inner.next()? {
            Ok(task) => self.local_pool.run_until(task).transpose(),
            Err(e) => Some(Err(e)),
        }
    }
}
