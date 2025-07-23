// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::future::BoxFuture;
use futures::stream::BoxStream;
use futures::{StreamExt, TryStreamExt, stream};
use vortex_array::ArrayRef;
use vortex_array::iter::ArrayIterator;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::work_queue::WorkStealingQueue;

pub(crate) type ArrayTask = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

/// A work-stealing iterator that supports dynamically adding tasks from task factories.
pub(crate) struct WorkStealingArrayIterator {
    queue: WorkStealingQueue<ArrayTask>,
    dtype: Arc<DType>,
    concurrency: usize,
    /// The internal stream of arrays for this worker iterator to process concurrently.
    stream: BoxStream<'static, VortexResult<ArrayRef>>,
}

impl WorkStealingArrayIterator {
    /// Creates a new `WorkStealingArrayIterator` with the provided tasks, data type, and
    /// concurrency level.
    ///
    /// The concurrency level determines how many tasks are processed concurrently by each worker.
    /// Higher concurrency results in the I/O for more tasks being kicked off ahead of time, but
    /// it also reduces the ability of workers to steal tasks from each other since concurrent
    /// tasks are allocated to a specific worker.
    pub(crate) fn new(
        queue: WorkStealingQueue<ArrayTask>,
        dtype: Arc<DType>,
        concurrency: usize,
    ) -> Self {
        let stream = Self::new_stream(queue.clone(), concurrency);
        Self {
            queue,
            dtype,
            concurrency,
            stream,
        }
    }

    fn new_stream(
        queue: WorkStealingQueue<ArrayTask>,
        concurrency: usize,
    ) -> BoxStream<'static, VortexResult<ArrayRef>> {
        // We set up a stream to pull from the inner iterator and process the futures concurrently.
        stream::iter(queue.new_iterator())
            .try_buffered(concurrency)
            .filter_map(|result| async move { result.transpose() })
            .boxed()
    }
}

/// Cloning a work-stealing iterator creates a new worker that can be driven independently by
/// another thread.
impl Clone for WorkStealingArrayIterator {
    fn clone(&self) -> Self {
        let stream = Self::new_stream(self.queue.clone(), self.concurrency);
        Self {
            queue: self.queue.clone(),
            dtype: self.dtype.clone(),
            concurrency: self.concurrency,
            stream,
        }
    }
}

impl ArrayIterator for WorkStealingArrayIterator {
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}

impl Iterator for WorkStealingArrayIterator {
    type Item = VortexResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        futures::executor::block_on(self.stream.next())
    }
}
