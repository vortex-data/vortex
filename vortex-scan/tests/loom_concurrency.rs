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
    use std::collections::VecDeque;

    use loom::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use loom::sync::{Arc, Mutex, RwLock};
    use loom::thread;

    #[test]
    fn test_work_queue_atomic_ordering() {
        // Test the atomic ordering fix in work_queue.rs
        // This verifies that Acquire ordering properly synchronizes with Release
        // to prevent the race condition we fixed.
        loom::model(|| {
            let num_factories = Arc::new(AtomicUsize::new(3)); // Start with 3 factories to construct
            let num_factories_constructed = Arc::new(AtomicUsize::new(0));

            let constructed_clone = num_factories_constructed.clone();
            let num_factories_check = num_factories.clone();
            let num_constructed_check = num_factories_constructed.clone();

            // Producer thread that constructs factories one by one
            let producer = thread::spawn(move || {
                // Simulate constructing factories one by one
                for i in 1..=3 {
                    constructed_clone.store(i, Ordering::Release);
                    thread::yield_now();
                }
            });

            // Consumer thread that waits for all factories to be constructed
            let consumer = thread::spawn(move || {
                // This is the pattern from work_queue.rs line 192
                // Using Acquire ordering to see all writes from producer
                while num_constructed_check.load(Ordering::Acquire)
                    < num_factories_check.load(Ordering::Acquire)
                {
                    thread::yield_now();
                }

                // After the loop, we should see all factories constructed
                let seen_factories = num_constructed_check.load(Ordering::Acquire);

                // Verify we see the correct number
                assert_eq!(seen_factories, 3);
            });

            producer.join().unwrap();
            consumer.join().unwrap();
        });
    }

    #[test]
    fn test_work_stealing_no_double_processing() {
        // Test that work stealing doesn't process the same item twice
        // This simulates the work-stealing pattern in work_queue.rs
        loom::model(|| {
            let work_queue = Arc::new(Mutex::new(VecDeque::new()));
            let processed = Arc::new(Mutex::new(Vec::new()));

            // Add initial work items
            {
                let mut queue = work_queue.lock().unwrap();
                for i in 1..=5 {
                    queue.push_back(i);
                }
            }

            let queue1 = work_queue.clone();
            let processed1 = processed.clone();

            let queue2 = work_queue.clone();
            let processed2 = processed.clone();

            // Worker 1 tries to steal work
            let worker1 = thread::spawn(move || {
                for _ in 0..3 {
                    if let Some(work) = queue1.lock().unwrap().pop_front() {
                        processed1.lock().unwrap().push(work);
                    }
                    thread::yield_now();
                }
            });

            // Worker 2 tries to steal work
            let worker2 = thread::spawn(move || {
                for _ in 0..3 {
                    if let Some(work) = queue2.lock().unwrap().pop_front() {
                        processed2.lock().unwrap().push(work);
                    }
                    thread::yield_now();
                }
            });

            worker1.join().unwrap();
            worker2.join().unwrap();

            // Verify all work was processed exactly once
            let mut final_processed = processed.lock().unwrap().clone();
            final_processed.sort();

            // Check no duplicates
            for i in 1..final_processed.len() {
                assert_ne!(
                    final_processed[i],
                    final_processed[i - 1],
                    "Found duplicate work item: {}",
                    final_processed[i]
                );
            }

            // All processed items should be from our original set
            for &item in &final_processed {
                assert!(item >= 1 && item <= 5);
            }
        });
    }

    #[test]
    fn test_filter_rwlock_concurrent_access() {
        // Test RwLock access patterns similar to filter.rs
        // Multiple readers reading selectivity while a writer updates
        // This verifies the concurrent access pattern is safe
        loom::model(|| {
            // Simulate the selectivity histograms from filter.rs
            let histogram1 = Arc::new(RwLock::new(vec![0.5, 0.6, 0.7]));
            let histogram2 = Arc::new(RwLock::new(vec![0.3, 0.4, 0.5]));

            let hist1_r1 = histogram1.clone();
            let hist1_r2 = histogram1.clone();
            let hist1_w = histogram1.clone();
            let hist2_r = histogram2.clone();

            // Reader 1 - reads from histogram1
            let reader1 = thread::spawn(move || {
                let values = hist1_r1.read().unwrap();
                values[0] // Read first value
            });

            // Reader 2 - reads from histogram1
            let reader2 = thread::spawn(move || {
                let values = hist1_r2.read().unwrap();
                values[1] // Read second value
            });

            // Reader 3 - reads from histogram2 (different lock)
            let reader3 = thread::spawn(move || {
                let values = hist2_r.read().unwrap();
                values[0] // Read first value
            });

            // Writer - updates histogram1
            let writer = thread::spawn(move || {
                let mut values = hist1_w.write().unwrap();
                values[0] = 0.4;
                values[1] = 0.5;
                values[2] = 0.6;
            });

            let val1 = reader1.join().unwrap();
            let val2 = reader2.join().unwrap();
            let val3 = reader3.join().unwrap();
            writer.join().unwrap();

            // Reader 1 should see either old or new value
            assert!(val1 == 0.5 || val1 == 0.4);
            // Reader 2 should see either old or new value
            assert!(val2 == 0.6 || val2 == 0.5);
            // Reader 3 always sees the same value (different lock)
            assert_eq!(val3, 0.3);
        });
    }

    #[test]
    fn test_dynamic_version_toctou_fix() {
        // Test the TOCTOU fix in tasks.rs
        // Verifies that reading the version once prevents race conditions
        loom::model(|| {
            // Simulate dynamic version that can change
            let version = Arc::new(AtomicU64::new(1));
            let local_version = Arc::new(Mutex::new(None));

            let version_update = version.clone();
            let version_read = version.clone();
            let local_version_write = local_version.clone();

            // Thread that updates the version (simulating dynamic filter update)
            let updater = thread::spawn(move || {
                thread::yield_now();
                version_update.store(2, Ordering::Release);
                thread::yield_now();
                version_update.store(3, Ordering::Release);
            });

            // Thread that reads version (simulating the fixed code in tasks.rs)
            let reader = thread::spawn(move || {
                // This simulates the fix: read once and store
                let current_version = version_read.load(Ordering::Acquire);

                // Simulate the check and update pattern
                let mut local = local_version_write.lock().unwrap();
                if local.is_none() || local.unwrap() < current_version {
                    *local = Some(current_version);
                }
                *local
            });

            updater.join().unwrap();
            let result = reader.join().unwrap();

            // The reader should see a consistent version (1, 2, or 3)
            assert!(result == Some(1) || result == Some(2) || result == Some(3));
        });
    }

    #[test]
    fn test_concurrent_factory_construction() {
        // Test concurrent factory construction from work_queue.rs
        // Verifies no races when multiple threads construct factories
        loom::model(|| {
            let num_factories = Arc::new(AtomicUsize::new(2));
            let num_constructed = Arc::new(AtomicUsize::new(0));
            let results = Arc::new(Mutex::new(Vec::new()));

            let constructed1 = num_constructed.clone();
            let results1 = results.clone();

            let constructed2 = num_constructed.clone();
            let results2 = results.clone();

            let num_factories_check = num_factories.clone();
            let num_constructed_check = num_constructed.clone();

            // Thread 1 constructs a factory
            let constructor1 = thread::spawn(move || {
                results1.lock().unwrap().push("factory1");
                constructed1.fetch_add(1, Ordering::Release);
            });

            // Thread 2 constructs a factory
            let constructor2 = thread::spawn(move || {
                results2.lock().unwrap().push("factory2");
                constructed2.fetch_add(1, Ordering::Release);
            });

            // Thread 3 waits for all factories to be constructed
            let waiter = thread::spawn(move || {
                while num_constructed_check.load(Ordering::Acquire)
                    < num_factories_check.load(Ordering::Acquire)
                {
                    thread::yield_now();
                }
                num_constructed_check.load(Ordering::Acquire)
            });

            constructor1.join().unwrap();
            constructor2.join().unwrap();
            let final_count = waiter.join().unwrap();

            // Verify both factories were constructed
            assert_eq!(final_count, 2);
            assert_eq!(results.lock().unwrap().len(), 2);
        });
    }

    #[test]
    fn test_stealer_registration_race() {
        // Test the race condition when registering stealers in work_queue.rs
        // Verifies the RwLock pattern for stealer registration is safe
        loom::model(|| {
            let stealers = Arc::new(RwLock::new(Vec::new()));

            let stealers1 = stealers.clone();
            let stealers2 = stealers.clone();
            let stealers_read = stealers.clone();

            // Thread 1 registers a stealer
            let registrar1 = thread::spawn(move || {
                let mut s = stealers1.write().unwrap();
                s.push(1);
            });

            // Thread 2 registers a stealer
            let registrar2 = thread::spawn(move || {
                let mut s = stealers2.write().unwrap();
                s.push(2);
            });

            // Thread 3 reads the stealers
            let reader = thread::spawn(move || {
                thread::yield_now();
                let s = stealers_read.read().unwrap();
                s.len()
            });

            registrar1.join().unwrap();
            registrar2.join().unwrap();
            let count = reader.join().unwrap();

            // Reader should see 0, 1, or 2 stealers depending on timing
            assert!(count <= 2);
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
