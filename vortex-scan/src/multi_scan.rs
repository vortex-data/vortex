// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::executor::LocalPool;
use futures::future::BoxFuture;
use vortex_error::VortexResult;

use crate::work_queue::{TaskFactory, WorkStealingIterator, WorkStealingQueue};

pub type ArrayFuture<T> = BoxFuture<'static, VortexResult<Option<T>>>;

/// A multi-scan for executing multiple scans concurrently across workers.
#[derive(Clone)]
pub struct MultiScan<T> {
    queue: WorkStealingQueue<ArrayFuture<T>>,
}

impl<T: 'static + Send> MultiScan<T> {
    /// Created with lazily constructed scan builders closures.
    pub fn new<I, F>(closures: I) -> Self
    where
        F: FnOnce() -> VortexResult<Vec<ArrayFuture<T>>> + 'static + Send + Sync,
        I: IntoIterator<Item = F>,
    {
        Self {
            queue: WorkStealingQueue::new(
                closures
                    .into_iter()
                    .map(|closure| Box::new(closure) as TaskFactory<ArrayFuture<T>>),
            ),
        }
    }

    pub fn new_iterator(self) -> MultiScanIterator<T> {
        MultiScanIterator {
            inner: self.queue.new_iterator(),
            local_pool: LocalPool::new(),
        }
    }
}

/// Scan iterator to participate in a `MultiScan`.
pub struct MultiScanIterator<T> {
    inner: WorkStealingIterator<ArrayFuture<T>>,
    local_pool: LocalPool,
}

impl<T> Clone for MultiScanIterator<T> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            local_pool: Default::default(),
        }
    }
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
