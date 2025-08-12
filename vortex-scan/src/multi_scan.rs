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
        loop {
            match self.inner.next()? {
                Ok(task) => match self.local_pool.run_until(task) {
                    // If the underlying future returns Ok(None) we have to keep going
                    // until we find the next present element or end of iterator.
                    Ok(Some(value)) => return Some(Ok(value)),
                    Ok(None) => continue,
                    Err(e) => return Some(Err(e)),
                },
                Err(e) => return Some(Err(e)),
            }
        }
    }
}
