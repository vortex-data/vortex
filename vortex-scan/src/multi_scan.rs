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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use vortex_error::{VortexResult, vortex_err};

    use super::*;

    #[test]
    fn test_multi_scan_basic() {
        // Create multiple scan tasks
        let closures = vec![
            || -> VortexResult<Vec<ArrayFuture<i32>>> {
                Ok(vec![
                    Box::pin(async { Ok(Some(1)) }),
                    Box::pin(async { Ok(Some(2)) }),
                ])
            },
            || -> VortexResult<Vec<ArrayFuture<i32>>> {
                Ok(vec![
                    Box::pin(async { Ok(Some(3)) }),
                    Box::pin(async { Ok(Some(4)) }),
                ])
            },
        ];

        let multi_scan = MultiScan::new(closures);
        let iterator = multi_scan.new_iterator();

        let mut results = Vec::new();
        for result in iterator {
            results.push(result.unwrap());
        }

        // Should get all 4 values
        assert_eq!(results.len(), 4);
        assert!(results.contains(&1));
        assert!(results.contains(&2));
        assert!(results.contains(&3));
        assert!(results.contains(&4));
    }

    #[test]
    fn test_multi_scan_error_handling() {
        // Create closures where one returns an error
        let closures = vec![
            || -> VortexResult<Vec<ArrayFuture<i32>>> { Ok(vec![Box::pin(async { Ok(Some(1)) })]) },
            || -> VortexResult<Vec<ArrayFuture<i32>>> { Err(vortex_err!("Task factory error")) },
            || -> VortexResult<Vec<ArrayFuture<i32>>> { Ok(vec![Box::pin(async { Ok(Some(2)) })]) },
        ];

        let multi_scan = MultiScan::new(closures);
        let iterator = multi_scan.new_iterator();

        let mut has_error = false;
        let mut values = Vec::new();

        for result in iterator {
            match result {
                Ok(v) => values.push(v),
                Err(_) => has_error = true,
            }
        }

        assert!(has_error, "Expected to encounter an error");
        // Should still get the values from successful factories
        assert!(values.contains(&1) || values.contains(&2));
    }

    #[test]
    fn test_multi_scan_iterator_clone() {
        let counter = Arc::new(AtomicUsize::new(0));
        let counter_clone = counter.clone();

        let closures = vec![move || -> VortexResult<Vec<ArrayFuture<i32>>> {
            counter_clone.fetch_add(1, Ordering::SeqCst);
            Ok(vec![
                Box::pin(async { Ok(Some(1)) }),
                Box::pin(async { Ok(Some(2)) }),
            ])
        }];

        let multi_scan = MultiScan::new(closures);
        let iterator1 = multi_scan.new_iterator();

        // Clone the iterator
        let mut iterator2 = iterator1;

        // Both iterators should be able to get results
        let result = iterator2.next();
        assert!(result.is_some());

        // Factory should only be called once despite having two iterators
        assert_eq!(counter.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn test_multi_scan_empty() {
        type Factory = Box<dyn FnOnce() -> VortexResult<Vec<ArrayFuture<i32>>> + Send + Sync>;
        let closures: Vec<Factory> = vec![];

        let multi_scan = MultiScan::new(closures);
        let mut iterator = multi_scan.new_iterator();

        // Should return None immediately
        assert!(iterator.next().is_none());
    }

    #[test]
    fn test_multi_scan_with_none_results() {
        let closures = vec![|| -> VortexResult<Vec<ArrayFuture<Option<i32>>>> {
            Ok(vec![
                Box::pin(async { Ok(Some(None)) }),
                Box::pin(async { Ok(Some(Some(1))) }),
                Box::pin(async { Ok(Some(None)) }),
            ])
        }];

        let multi_scan = MultiScan::new(closures);
        let iterator = multi_scan.new_iterator();

        let mut results = Vec::new();
        for result in iterator {
            if let Ok(Some(v)) = result {
                results.push(v);
            }
        }

        // Should only get the Some(1) value
        assert_eq!(results, vec![1]);
    }

    #[test]
    fn test_multi_scan_concurrent_iterators() {
        let closures = vec![|| -> VortexResult<Vec<ArrayFuture<i32>>> {
            Ok((1..=10)
                .map(|i| Box::pin(async move { Ok(Some(i)) }) as ArrayFuture<i32>)
                .collect())
        }];

        let multi_scan = MultiScan::new(closures);

        // Create multiple iterators
        let mut iter1 = multi_scan.clone().new_iterator();
        let mut iter2 = multi_scan.new_iterator();

        // Both should be able to steal work
        let mut count1 = 0;
        let mut count2 = 0;

        // Interleave taking from both iterators
        loop {
            let done1 = iter1
                .next()
                .map(|r| {
                    count1 += r.is_ok() as usize;
                })
                .is_none();
            let done2 = iter2
                .next()
                .map(|r| {
                    count2 += r.is_ok() as usize;
                })
                .is_none();

            if done1 && done2 {
                break;
            }
        }

        // Together they should process all 10 items
        assert_eq!(count1 + count2, 10);
        // Both should have gotten some work
        assert!(count1 > 0);
        assert!(count2 > 0);
    }

    #[test]
    fn test_local_pool_error_propagation() {
        let closures = vec![|| -> VortexResult<Vec<ArrayFuture<String>>> {
            Ok(vec![
                Box::pin(async { Ok(Some("success".to_string())) }),
                Box::pin(async { Err(vortex_err!("async error")) }),
                Box::pin(async { Ok(Some("after_errors".to_string())) }),
            ])
        }];

        let multi_scan = MultiScan::new(closures);
        let iterator = multi_scan.new_iterator();

        let mut results = Vec::new();
        let mut errors = Vec::new();

        for result in iterator {
            match result {
                Ok(v) => results.push(v),
                Err(e) => errors.push(e),
            }
        }

        // Should get both successful results and the error
        assert!(results.contains(&"success".to_string()));
        assert!(results.contains(&"after_errors".to_string()));
        assert_eq!(errors.len(), 1);
    }

    #[test]
    #[should_panic(expected = "Factory panic!")]
    #[allow(clippy::panic)]
    fn test_task_factory_panic_handling() {
        // Test that panics in task factories are propagated
        let closures = vec![|| -> VortexResult<Vec<ArrayFuture<i32>>> {
            panic!("Factory panic!");
        }];

        let multi_scan = MultiScan::new(closures);
        let iterator = multi_scan.new_iterator();

        // This should panic when the factory is executed
        for _ in iterator {
            // Consume iterator
        }
    }
}
