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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow_array::Int32Array;
    use vortex_array::ArrayRef;
    use vortex_array::arrow::FromArrowArray;
    use vortex_dtype::{DType, Nullability, PType};
    use vortex_error::vortex_err;

    use super::*;
    use crate::work_queue::{TaskFactory, WorkStealingQueue};

    fn create_test_array(value: i32) -> ArrayRef {
        let arrow_array = Int32Array::from(vec![value]);
        ArrayRef::from_arrow(&arrow_array, true)
    }

    #[test]
    fn test_basic_iteration() {
        let dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));

        let tasks: Vec<TaskFactory<ArrayTask>> = vec![
            Box::new(|| {
                Ok(vec![
                    Box::pin(async { Ok(Some(create_test_array(1))) }),
                    Box::pin(async { Ok(Some(create_test_array(2))) }),
                ])
            }),
            Box::new(|| Ok(vec![Box::pin(async { Ok(Some(create_test_array(3))) })])),
        ];

        let queue = WorkStealingQueue::new(tasks);
        let iterator = WorkStealingArrayIterator::new(queue, dtype, 1);

        let mut count = 0;
        for result in iterator {
            assert!(result.is_ok());
            count += 1;
        }

        assert_eq!(count, 3);
    }

    #[test]
    fn test_concurrent_processing() {
        let dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let counter = Arc::new(AtomicUsize::new(0));

        let tasks: Vec<TaskFactory<ArrayTask>> = (0..10)
            .map(|i| {
                let counter = counter.clone();
                Box::new(move || {
                    Ok(vec![Box::pin(async move {
                        counter.fetch_add(1, Ordering::SeqCst);
                        Ok(Some(create_test_array(i)))
                    }) as ArrayTask])
                }) as TaskFactory<ArrayTask>
            })
            .collect();

        let queue = WorkStealingQueue::new(tasks);
        // Test with higher concurrency
        let iterator = WorkStealingArrayIterator::new(queue, dtype, 4);

        let mut results = Vec::new();
        for result in iterator {
            results.push(result);
        }

        assert_eq!(results.len(), 10);
        assert_eq!(counter.load(Ordering::SeqCst), 10);
    }

    #[test]
    fn test_clone_creates_new_worker() {
        let dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));

        let tasks: Vec<TaskFactory<ArrayTask>> = vec![Box::new(|| {
            Ok((0..10)
                .map(|i| Box::pin(async move { Ok(Some(create_test_array(i))) }) as ArrayTask)
                .collect())
        })];

        let queue = WorkStealingQueue::new(tasks);
        let mut iterator1 = WorkStealingArrayIterator::new(queue, dtype, 2);
        let mut iterator2 = iterator1.clone();

        // Both iterators should share the same dtype
        assert_eq!(iterator1.dtype(), iterator2.dtype());

        // Both should be able to pull work
        let mut count1 = 0;
        let mut count2 = 0;

        // Interleave pulling from both iterators
        loop {
            let done1 = iterator1.next().map(|_| count1 += 1).is_none();
            let done2 = iterator2.next().map(|_| count2 += 1).is_none();

            if done1 && done2 {
                break;
            }
        }

        // Together they should process all 10 items
        assert_eq!(count1 + count2, 10);
        // Both should have gotten some work (work stealing)
        assert!(count1 > 0);
        assert!(count2 > 0);
    }

    #[test]
    fn test_error_propagation() {
        let dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));

        let tasks: Vec<TaskFactory<ArrayTask>> = vec![Box::new(|| {
            Ok(vec![
                Box::pin(async { Ok(Some(create_test_array(1))) }),
                Box::pin(async { Err(vortex_err!("test error")) }),
                Box::pin(async { Ok(Some(create_test_array(2))) }),
            ])
        })];

        let queue = WorkStealingQueue::new(tasks);
        let iterator = WorkStealingArrayIterator::new(queue, dtype, 1);

        let mut successes = 0;
        let mut errors = 0;

        for result in iterator {
            match result {
                Ok(_) => successes += 1,
                Err(_) => errors += 1,
            }
        }

        assert_eq!(successes, 2);
        assert_eq!(errors, 1);
    }

    #[test]
    fn test_filter_none_results() {
        let dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));

        let tasks: Vec<TaskFactory<ArrayTask>> = vec![Box::new(|| {
            Ok(vec![
                Box::pin(async { Ok(None) }),
                Box::pin(async { Ok(Some(create_test_array(1))) }),
                Box::pin(async { Ok(None) }),
                Box::pin(async { Ok(Some(create_test_array(2))) }),
            ])
        })];

        let queue = WorkStealingQueue::new(tasks);
        let iterator = WorkStealingArrayIterator::new(queue, dtype, 1);

        let mut count = 0;
        for result in iterator {
            assert!(result.is_ok());
            count += 1;
        }

        // Should only get the Some results
        assert_eq!(count, 2);
    }

    #[test]
    fn test_empty_queue() {
        let dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));
        let tasks: Vec<TaskFactory<ArrayTask>> = vec![];

        let queue = WorkStealingQueue::new(tasks);
        let mut iterator = WorkStealingArrayIterator::new(queue, dtype, 1);

        // Should return None immediately
        assert!(iterator.next().is_none());
    }

    #[test]
    fn test_different_concurrency_levels() {
        let dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));

        // Test different concurrency levels
        for concurrency in [1, 2, 4] {
            let tasks: Vec<TaskFactory<ArrayTask>> = vec![Box::new(move || {
                Ok((0..8)
                    .map(|i| Box::pin(async move { Ok(Some(create_test_array(i))) }) as ArrayTask)
                    .collect())
            })];

            let queue = WorkStealingQueue::new(tasks);
            let iterator = WorkStealingArrayIterator::new(queue, dtype.clone(), concurrency);

            let mut count = 0;
            for result in iterator {
                assert!(result.is_ok());
                count += 1;
            }

            // Should process all 8 items regardless of concurrency
            assert_eq!(count, 8, "Failed for concurrency={}", concurrency);
        }
    }

    #[test]
    fn test_factory_error() {
        let dtype = Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable));

        let tasks: Vec<TaskFactory<ArrayTask>> = vec![
            Box::new(|| Ok(vec![Box::pin(async { Ok(Some(create_test_array(1))) })])),
            Box::new(|| Err(vortex_err!("Factory construction error"))),
            Box::new(|| Ok(vec![Box::pin(async { Ok(Some(create_test_array(2))) })])),
        ];

        let queue = WorkStealingQueue::new(tasks);
        let iterator = WorkStealingArrayIterator::new(queue, dtype, 1);

        let mut successes = 0;
        let mut errors = 0;

        for result in iterator {
            match result {
                Ok(_) => successes += 1,
                Err(_) => errors += 1,
            }
        }

        // Factory errors are propagated as iterator errors
        assert_eq!(successes, 2);
        assert_eq!(errors, 1);
    }
}
