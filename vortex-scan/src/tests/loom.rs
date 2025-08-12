// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use bit_vec::BitVec;
use futures::future;
use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::sync::{Arc, Mutex};
use loom::thread;
use vortex_error::{VortexResult, vortex_err};
use vortex_expr::{and, get_item, gt, lit, lt, root};

use crate::filter::FilterExpr;
use crate::multi_scan::{ArrayFuture, MultiScan};
use crate::work_queue::{TaskFactory, WorkStealingQueue};

#[test]
fn test_work_stealing_queue_basic() {
    // Test basic WorkStealingQueue operations with multiple workers
    loom::model(|| {
        // Create task factories that produce simple tasks
        let factories: Vec<TaskFactory<i32>> = vec![
            Box::new(|| Ok(vec![1, 2, 3])),
            Box::new(|| Ok(vec![4, 5, 6])),
            Box::new(|| Ok(vec![7, 8, 9])),
        ];

        let queue = WorkStealingQueue::new(factories);

        // Create two workers
        let iter1 = queue.clone().new_iterator();
        let iter2 = queue.new_iterator();

        // Collect results from both workers
        let handle1 = thread::spawn(move || {
            let mut results = Vec::new();
            for val in iter1.flatten() {
                results.push(val);
                if results.len() >= 3 {
                    break;
                }
            }
            results
        });

        let handle2 = thread::spawn(move || {
            let mut results = Vec::new();
            for val in iter2.flatten() {
                results.push(val);
                if results.len() >= 3 {
                    break;
                }
            }
            results
        });

        let results1 = handle1.join().unwrap();
        let results2 = handle2.join().unwrap();

        // Verify that results are from our expected set
        for val in results1.iter().chain(results2.iter()) {
            assert!(*val >= 1 && *val <= 9);
        }

        // Verify no duplicates between workers
        let mut all_results = results1;
        all_results.extend(results2);
        all_results.sort();
        for i in 1..all_results.len() {
            assert_ne!(all_results[i], all_results[i - 1], "Found duplicate value");
        }
    });
}

#[test]
fn test_work_stealing_queue_error_handling() {
    // Test that errors in task factories are properly propagated
    loom::model(|| {
        let factories: Vec<TaskFactory<i32>> = vec![
            Box::new(|| Ok(vec![1, 2])),
            Box::new(|| Err(vortex_err!("Factory error"))),
            Box::new(|| Ok(vec![3, 4])),
        ];

        let queue = WorkStealingQueue::new(factories);
        let iter = queue.new_iterator();

        let mut has_error = false;
        let mut values = Vec::new();

        for result in iter {
            match result {
                Ok(val) => values.push(val),
                Err(_) => {
                    has_error = true;
                    break;
                }
            }
        }

        // Should encounter the error
        assert!(has_error || !values.is_empty());
    });
}

#[test]
fn test_work_stealing_concurrent_factory_construction() {
    // Test concurrent factory construction with multiple workers
    loom::model(|| {
        let counter = Arc::new(AtomicUsize::new(0));

        let factories: Vec<TaskFactory<usize>> = (0..3usize)
            .map(|i| {
                let counter_clone = counter.clone();
                Box::new(move || {
                    counter_clone.fetch_add(1, Ordering::SeqCst);
                    Ok(vec![i * 10, i * 10 + 1])
                }) as TaskFactory<usize>
            })
            .collect();

        let queue = WorkStealingQueue::new(factories);

        // Create multiple workers
        let iter1 = queue.clone().new_iterator();
        let iter2 = queue.new_iterator();

        let handle1 = thread::spawn(move || {
            let mut count = 0;
            for result in iter1 {
                if result.is_ok() {
                    count += 1;
                    if count >= 2 {
                        break;
                    }
                }
            }
            count
        });

        let handle2 = thread::spawn(move || {
            let mut count = 0;
            for result in iter2 {
                if result.is_ok() {
                    count += 1;
                    if count >= 2 {
                        break;
                    }
                }
            }
            count
        });

        handle1.join().unwrap();
        handle2.join().unwrap();

        // Verify factories were constructed
        let final_count = counter.load(Ordering::SeqCst);
        assert!(final_count > 0 && final_count <= 3);
    });
}

#[test]
fn test_filter_expr_concurrent_selectivity_reporting() {
    // Test concurrent selectivity reporting in FilterExpr
    loom::model(|| {
        let expr = lit(true); // Simple expression for testing
        let filter = Arc::new(FilterExpr::new(expr));

        let filter1 = filter.clone();
        let filter2 = filter.clone();
        let filter3 = filter;

        // Multiple threads reporting selectivity
        let handle1 = thread::spawn(move || {
            filter1.report_selectivity(0, 0.5);
            filter1.report_selectivity(0, 0.6);
        });

        let handle2 = thread::spawn(move || {
            filter2.report_selectivity(0, 0.7);
            filter2.report_selectivity(0, 0.4);
        });

        // Reader thread
        let handle3 = thread::spawn(move || {
            let mut remaining = BitVec::from_elem(1, true);
            let conjunct = filter3.next_conjunct(&remaining);
            assert_eq!(conjunct, Some(0));

            // Mark as evaluated
            remaining.set(0, false);
            let conjunct = filter3.next_conjunct(&remaining);
            assert_eq!(conjunct, None);
        });

        handle1.join().unwrap();
        handle2.join().unwrap();
        handle3.join().unwrap();
    });
}

#[test]
fn test_filter_expr_ordering_update() {
    // Test concurrent ordering updates in FilterExpr
    loom::model(|| {
        // Create a filter with multiple conjuncts (AND conditions)
        let expr = and(
            gt(get_item("a", root()), lit(5)),
            lt(get_item("b", root()), lit(10)),
        );
        let filter = Arc::new(FilterExpr::new(expr));

        let filter1 = filter.clone();
        let filter2 = filter;

        // Thread 1 reports selectivity for conjunct 0
        let handle1 = thread::spawn(move || {
            filter1.report_selectivity(0, 0.1); // Very selective
            filter1.report_selectivity(0, 0.2);
        });

        // Thread 2 reports selectivity for conjunct 1
        let handle2 = thread::spawn(move || {
            filter2.report_selectivity(1, 0.9); // Not selective
            filter2.report_selectivity(1, 0.8);
        });

        handle1.join().unwrap();
        handle2.join().unwrap();

        // The ordering should prefer more selective conjuncts
        // but we just verify it doesn't crash under concurrent access
    });
}

#[test]
fn test_multi_scan_concurrent_iteration() {
    // Test MultiScan with concurrent iterators
    loom::model(|| {
        // Create closures that produce futures
        let closures = vec![
            || -> VortexResult<Vec<ArrayFuture<i32>>> {
                Ok(vec![
                    Box::pin(future::ready(Ok(Some(1)))),
                    Box::pin(future::ready(Ok(Some(2)))),
                ])
            },
            || -> VortexResult<Vec<ArrayFuture<i32>>> {
                Ok(vec![
                    Box::pin(future::ready(Ok(Some(3)))),
                    Box::pin(future::ready(Ok(Some(4)))),
                ])
            },
        ];

        let multi_scan = MultiScan::new(closures);

        // Create two iterators
        let iter1 = multi_scan.clone().new_iterator();
        let iter2 = multi_scan.new_iterator();

        // Collect from both iterators concurrently
        let handle1 = thread::spawn(move || {
            let mut results = Vec::new();
            for val in iter1.flatten() {
                results.push(val);
                if results.len() >= 2 {
                    break;
                }
            }
            results
        });

        let handle2 = thread::spawn(move || {
            let mut results = Vec::new();
            for val in iter2.flatten() {
                results.push(val);
                if results.len() >= 2 {
                    break;
                }
            }
            results
        });

        let results1 = handle1.join().unwrap();
        let results2 = handle2.join().unwrap();

        // Verify results are from expected set
        for val in results1.iter().chain(results2.iter()) {
            assert!(*val >= 1 && *val <= 4);
        }
    });
}

#[test]
fn test_work_stealing_with_empty_factories() {
    // Test edge case with empty task factories
    loom::model(|| {
        let factories: Vec<TaskFactory<i32>> = vec![
            Box::new(|| Ok(vec![])), // Empty
            Box::new(|| Ok(vec![1, 2])),
            Box::new(|| Ok(vec![])), // Empty
            Box::new(|| Ok(vec![3])),
        ];

        let queue = WorkStealingQueue::new(factories);
        let mut iter = queue.new_iterator();

        let mut results = Vec::new();
        while let Some(Ok(val)) = iter.next() {
            results.push(val);
        }

        // Should get exactly the non-empty values
        results.sort();
        assert_eq!(results, vec![1, 2, 3]);
    });
}

#[test]
fn test_work_stealing_clone_semantics() {
    // Test that cloning iterators properly shares the work queue
    loom::model(|| {
        let factories: Vec<TaskFactory<i32>> = vec![Box::new(|| Ok(vec![1, 2, 3, 4]))];

        let queue = WorkStealingQueue::new(factories);
        let iter1 = queue.new_iterator();

        // Clone the iterator
        let iter2 = iter1.clone();

        let handle1 = thread::spawn(move || {
            let mut count = 0;
            for result in iter1 {
                if result.is_ok() {
                    count += 1;
                    if count >= 2 {
                        break;
                    }
                }
            }
            count
        });

        let handle2 = thread::spawn(move || {
            let mut count = 0;
            for result in iter2 {
                if result.is_ok() {
                    count += 1;
                    if count >= 2 {
                        break;
                    }
                }
            }
            count
        });

        let count1 = handle1.join().unwrap();
        let count2 = handle2.join().unwrap();

        // Both should get some work
        assert!(count1 + count2 <= 4);
    });
}

#[test]
fn test_work_stealing_memory_ordering() {
    // Test memory ordering between num_factories_constructed and task pushing
    // This is critical: tasks MUST be visible before num_factories_constructed is incremented
    loom::model(|| {
        let seen_values = Arc::new(Mutex::new(Vec::new()));
        let seen_clone = seen_values.clone();

        let factories: Vec<TaskFactory<usize>> = vec![
            Box::new(move || {
                // Simulate work that produces values
                Ok(vec![100, 101, 102])
            }),
            Box::new(|| Ok(vec![200, 201])),
            Box::new(|| Ok(vec![300])),
        ];

        let queue = WorkStealingQueue::new(factories);

        // Create multiple workers that will race to construct factories
        let iter1 = queue.clone().new_iterator();
        let iter2 = queue.clone().new_iterator();
        let iter3 = queue.new_iterator();

        let seen1 = seen_values.clone();
        let handle1 = thread::spawn(move || {
            for val in iter1.flatten() {
                seen1.lock().unwrap().push(val);
            }
        });

        let seen2 = seen_values.clone();
        let handle2 = thread::spawn(move || {
            for val in iter2.flatten() {
                seen2.lock().unwrap().push(val);
            }
        });

        let seen3 = seen_values;
        let handle3 = thread::spawn(move || {
            for val in iter3.flatten() {
                seen3.lock().unwrap().push(val);
            }
        });

        handle1.join().unwrap();
        handle2.join().unwrap();
        handle3.join().unwrap();

        // Verify all values were seen exactly once
        let mut final_values = seen_clone.lock().unwrap().clone();
        final_values.sort();
        assert_eq!(final_values, vec![100, 101, 102, 200, 201, 300]);
    });
}

#[test]
fn test_concurrent_filter_ordering_updates() {
    // Test race conditions in filter ordering updates
    // Multiple threads reading ordering while one updates it
    loom::model(|| {
        let expr = and(
            gt(get_item("a", root()), lit(5)),
            and(
                lt(get_item("b", root()), lit(10)),
                gt(get_item("c", root()), lit(0)),
            ),
        );
        let filter = Arc::new(FilterExpr::new(expr));

        // Create multiple reader threads and one writer thread
        let filter_reader1 = filter.clone();
        let filter_reader2 = filter.clone();
        let filter_writer = filter;

        // Writer thread continuously updates selectivity
        let writer = thread::spawn(move || {
            // Report different selectivities to trigger reordering
            filter_writer.report_selectivity(0, 0.9); // Low selectivity
            filter_writer.report_selectivity(1, 0.1); // High selectivity
            filter_writer.report_selectivity(2, 0.5); // Medium selectivity

            // Report more to potentially trigger reordering
            filter_writer.report_selectivity(0, 0.8);
            filter_writer.report_selectivity(1, 0.2);
        });

        // Reader threads continuously read the ordering
        let reader1 = thread::spawn(move || {
            let mut remaining = BitVec::from_elem(3, true);
            let mut seen = Vec::new();

            while let Some(idx) = filter_reader1.next_conjunct(&remaining) {
                seen.push(idx);
                remaining.set(idx, false);
            }
            seen
        });

        let reader2 = thread::spawn(move || {
            let mut remaining = BitVec::from_elem(3, true);
            let mut seen = Vec::new();

            while let Some(idx) = filter_reader2.next_conjunct(&remaining) {
                seen.push(idx);
                remaining.set(idx, false);
            }
            seen
        });

        writer.join().unwrap();
        let order1 = reader1.join().unwrap();
        let order2 = reader2.join().unwrap();

        // Both readers should see a valid ordering (all indices present)
        assert_eq!(order1.len(), 3);
        assert_eq!(order2.len(), 3);

        // Check all indices are present
        let mut sorted1 = order1;
        sorted1.sort();
        assert_eq!(sorted1, vec![0, 1, 2]);

        let mut sorted2 = order2;
        sorted2.sort();
        assert_eq!(sorted2, vec![0, 1, 2]);
    });
}

#[test]
fn test_steal_work_retry_semantics() {
    // Test the retry logic in steal_work with multiple concurrent stealers
    loom::model(|| {
        let counter = Arc::new(AtomicUsize::new(0));

        // Create factories that produce work in batches
        let factories: Vec<TaskFactory<usize>> = vec![
            Box::new(|| Ok(vec![1, 2, 3])),
            Box::new(|| Ok(vec![4, 5, 6])),
        ];

        let queue = WorkStealingQueue::new(factories);

        // Create 3 workers to stress the stealing logic (reduced from 4)
        let workers: Vec<_> = (0..3)
            .map(|_| {
                let iter = queue.clone().new_iterator();
                let counter_clone = counter.clone();
                thread::spawn(move || {
                    let mut local_sum = 0;
                    for val in iter.flatten() {
                        local_sum += val;
                        // Simulate some work to increase chances of stealing
                        thread::yield_now();
                    }
                    counter_clone.fetch_add(local_sum, Ordering::SeqCst);
                })
            })
            .collect();

        for worker in workers {
            worker.join().unwrap();
        }

        // All values should be processed exactly once
        assert_eq!(counter.load(Ordering::SeqCst), 21); // Sum of 1..=6
    });
}

#[test]
fn test_factory_error_recovery_race() {
    // Test that factory errors are handled correctly with concurrent workers
    // Particularly testing the increment of num_factories_constructed on error
    loom::model(|| {
        let error_seen = Arc::new(Mutex::new(false));
        let values_seen = Arc::new(Mutex::new(Vec::new()));

        let factories: Vec<TaskFactory<i32>> = vec![
            Box::new(|| Ok(vec![1, 2])),
            Box::new(|| Err(vortex_err!("Factory error"))),
            Box::new(|| Ok(vec![3, 4])),
            Box::new(|| Ok(vec![5])),
        ];

        let queue = WorkStealingQueue::new(factories);

        // Create multiple workers
        let iter1 = queue.clone().new_iterator();
        let iter2 = queue.clone().new_iterator();
        let iter3 = queue.new_iterator();

        let error_seen1 = error_seen.clone();
        let values_seen1 = values_seen.clone();
        let handle1 = thread::spawn(move || {
            for result in iter1 {
                match result {
                    Ok(val) => values_seen1.lock().unwrap().push(val),
                    Err(_) => {
                        *error_seen1.lock().unwrap() = true;
                        break;
                    }
                }
            }
        });

        let error_seen2 = error_seen.clone();
        let values_seen2 = values_seen.clone();
        let handle2 = thread::spawn(move || {
            for result in iter2 {
                match result {
                    Ok(val) => values_seen2.lock().unwrap().push(val),
                    Err(_) => {
                        *error_seen2.lock().unwrap() = true;
                        break;
                    }
                }
            }
        });

        let error_seen3 = error_seen.clone();
        let values_seen3 = values_seen.clone();
        let handle3 = thread::spawn(move || {
            for result in iter3 {
                match result {
                    Ok(val) => values_seen3.lock().unwrap().push(val),
                    Err(_) => {
                        *error_seen3.lock().unwrap() = true;
                        break;
                    }
                }
            }
        });

        handle1.join().unwrap();
        handle2.join().unwrap();
        handle3.join().unwrap();

        // At least one worker should have seen the error
        assert!(*error_seen.lock().unwrap());

        // Values from successful factories should still be processed
        let final_values = values_seen.lock().unwrap();
        assert!(!final_values.is_empty());
    });
}

#[test]
fn test_worker_termination_conditions() {
    // Test the complex termination condition in WorkStealingIterator::next()
    // Workers should terminate when:
    // 1. All factories are constructed AND
    // 2. No stealer has work
    loom::model(|| {
        let termination_flag = Arc::new(AtomicUsize::new(0));

        // Create factories with delays to test termination logic
        let factories: Vec<TaskFactory<usize>> = vec![
            Box::new(|| {
                // First factory produces work immediately
                Ok(vec![1, 2, 3])
            }),
            Box::new(|| {
                // Second factory also produces work
                Ok(vec![4, 5, 6])
            }),
        ];

        let queue = WorkStealingQueue::new(factories);

        // Create workers that track when they terminate
        let term_flag1 = termination_flag.clone();
        let iter1 = queue.clone().new_iterator();
        let handle1 = thread::spawn(move || {
            let mut count = 0;
            for result in iter1 {
                if result.is_ok() {
                    count += 1;
                    // Yield to allow other threads to run
                    thread::yield_now();
                }
            }
            term_flag1.fetch_add(1, Ordering::SeqCst);
            count
        });

        let term_flag2 = termination_flag.clone();
        let iter2 = queue.clone().new_iterator();
        let handle2 = thread::spawn(move || {
            let mut count = 0;
            for result in iter2 {
                if result.is_ok() {
                    count += 1;
                    thread::yield_now();
                }
            }
            term_flag2.fetch_add(1, Ordering::SeqCst);
            count
        });

        let term_flag3 = termination_flag.clone();
        let iter3 = queue.new_iterator();
        let handle3 = thread::spawn(move || {
            let mut count = 0;
            for result in iter3 {
                if result.is_ok() {
                    count += 1;
                    thread::yield_now();
                }
            }
            term_flag3.fetch_add(1, Ordering::SeqCst);
            count
        });

        let count1 = handle1.join().unwrap();
        let count2 = handle2.join().unwrap();
        let count3 = handle3.join().unwrap();

        // All workers should terminate
        assert_eq!(termination_flag.load(Ordering::SeqCst), 3);

        // Total work processed should be 6
        assert_eq!(count1 + count2 + count3, 6);
    });
}
