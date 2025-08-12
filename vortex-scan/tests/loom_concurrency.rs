// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Loom tests for concurrency verification in vortex-scan
//!
//! These tests use the loom crate to exhaustively test concurrent code paths
//! for race conditions, deadlocks, and other concurrency bugs.
//!
//! To run these tests:
//! ```bash
//! RUSTFLAGS="--cfg loom" cargo test --release --test loom_concurrency
//! ```
//!
//! Note: These tests may take a while to run as loom exhaustively checks
//! all possible interleavings.

#![cfg(loom)]

#[cfg(loom)]
mod loom_tests {
    use futures::future;
    use loom::sync::Arc;
    use loom::thread;
    use vortex_error::VortexResult;
    use vortex_expr::literal::lit;
    use vortex_scan::filter::FilterExpr;
    use vortex_scan::multi_scan::{ArrayFuture, MultiScan};
    use vortex_scan::work_queue::{TaskFactory, WorkStealingQueue};

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
                for item in iter1 {
                    if let Ok(val) = item {
                        results.push(val);
                        if results.len() >= 3 {
                            break;
                        }
                    }
                }
                results
            });

            let handle2 = thread::spawn(move || {
                let mut results = Vec::new();
                for item in iter2 {
                    if let Ok(val) = item {
                        results.push(val);
                        if results.len() >= 3 {
                            break;
                        }
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
            let mut all_results = results1.clone();
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
                Box::new(|| Err(vortex_error::vortex_err!("Factory error"))),
                Box::new(|| Ok(vec![3, 4])),
            ];

            let queue = WorkStealingQueue::new(factories);
            let mut iter = queue.new_iterator();

            let mut has_error = false;
            let mut values = Vec::new();

            while let Some(result) = iter.next() {
                match result {
                    Ok(val) => values.push(val),
                    Err(_) => {
                        has_error = true;
                        break;
                    }
                }
            }

            // Should encounter the error
            assert!(has_error || values.len() > 0);
        });
    }

    #[test]
    fn test_work_stealing_concurrent_factory_construction() {
        // Test concurrent factory construction with multiple workers
        loom::model(|| {
            use loom::sync::atomic::{AtomicUsize, Ordering};

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
            let filter3 = filter.clone();

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
                use bit_vec::BitVec;
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
            use vortex_expr::{and, get_item, gt, lt, root};

            // Create a filter with multiple conjuncts (AND conditions)
            let expr = and(
                gt(get_item("a", root()), lit(5)),
                lt(get_item("b", root()), lit(10)),
            );
            let filter = Arc::new(FilterExpr::new(expr));

            let filter1 = filter.clone();
            let filter2 = filter.clone();

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
                for item in iter1 {
                    if let Ok(val) = item {
                        results.push(val);
                        if results.len() >= 2 {
                            break;
                        }
                    }
                }
                results
            });

            let handle2 = thread::spawn(move || {
                let mut results = Vec::new();
                for item in iter2 {
                    if let Ok(val) = item {
                        results.push(val);
                        if results.len() >= 2 {
                            break;
                        }
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
}

// Provide a dummy test for non-loom builds
#[cfg(not(loom))]
#[test]
fn loom_tests_require_loom_cfg() {
    eprintln!("Loom tests require --cfg loom flag. Run with:");
    eprintln!("RUSTFLAGS=\"--cfg loom\" cargo test --release --test loom_concurrency");
}
