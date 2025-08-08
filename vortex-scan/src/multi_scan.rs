// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt, stream};
use vortex_error::VortexResult;

use crate::work_queue::{TaskFactory, WorkStealingQueue};

pub type ArrayFuture<T> = BoxFuture<'static, VortexResult<Option<T>>>;

/// A multi-scan for executing multiple scans concurrently across workers.
#[derive(Clone)]
pub struct MultiScan<T> {
    queue: WorkStealingQueue<ArrayFuture<T>>,
}

impl<T: 'static + Send + Sync> MultiScan<T> {
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
        Self::new_iterator_with_concurrency(self, 4) // Default concurrency of 4
    }

    pub fn new_iterator_with_concurrency(self, concurrency: usize) -> MultiScanIterator<T> {
        let stream = MultiScanIterator::new_stream(self.queue.clone(), concurrency);
        MultiScanIterator {
            queue: self.queue,
            concurrency,
            stream,
        }
    }
}

/// Scan iterator to participate in a `MultiScan`.
pub struct MultiScanIterator<T> {
    queue: WorkStealingQueue<ArrayFuture<T>>,
    concurrency: usize,
    stream: BoxStream<'static, VortexResult<T>>,
}

impl<T: Send + Sync + 'static> MultiScanIterator<T> {
    fn new_stream(
        queue: WorkStealingQueue<ArrayFuture<T>>,
        concurrency: usize,
    ) -> BoxStream<'static, VortexResult<T>> {
        stream::iter(queue.new_iterator())
            .try_buffered(concurrency)
            .filter_map(|result| async move { result.transpose() })
            .boxed()
    }
}

impl<T: Send + Sync + 'static> Clone for MultiScanIterator<T> {
    fn clone(&self) -> Self {
        let stream = Self::new_stream(self.queue.clone(), self.concurrency);
        Self {
            queue: self.queue.clone(),
            concurrency: self.concurrency,
            stream,
        }
    }
}

impl<T: Send + Sync + 'static> Iterator for MultiScanIterator<T> {
    type Item = VortexResult<T>;

    fn next(&mut self) -> Option<Self::Item> {
        futures::executor::block_on(self.stream.next())
    }
}
