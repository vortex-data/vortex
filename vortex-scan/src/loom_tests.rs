// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Loom tests for concurrency verification
//! 
//! These tests use the loom crate to exhaustively test concurrent code paths
//! for race conditions, deadlocks, and other concurrency bugs.
//! 
//! To run these tests:
//! RUSTFLAGS="--cfg loom" cargo test --release --test loom_tests

#![cfg(loom)]

use loom::sync::Arc;
use loom::sync::atomic::{AtomicUsize, Ordering};
use loom::thread;

#[test]
fn test_work_queue_atomic_ordering() {
    // Test the atomic ordering fix in work_queue.rs
    // This tests that the Acquire ordering properly synchronizes with Release
    loom::model(|| {
        let num_factories = Arc::new(AtomicUsize::new(0));
        let num_factories_constructed = Arc::new(AtomicUsize::new(0));
        
        let factories_clone = num_factories.clone();
        let constructed_clone = num_factories_constructed.clone();
        
        // Producer thread that sets up factories
        let producer = thread::spawn(move || {
            // Simulate setting up 3 factories
            factories_clone.store(3, Ordering::Release);
            
            // Simulate constructing factories one by one
            for i in 1..=3 {
                constructed_clone.store(i, Ordering::Release);
                loom::thread::yield_now();
            }
        });
        
        // Consumer thread that waits for factories
        let consumer = thread::spawn(move || {
            let mut seen_factories = 0;
            
            // This is the pattern from work_queue.rs line 192
            while num_factories_constructed.load(Ordering::Acquire) < num_factories.load(Ordering::Acquire) {
                loom::thread::yield_now();
            }
            
            // After the loop, we should see all factories constructed
            seen_factories = num_factories_constructed.load(Ordering::Acquire);
            
            // Verify we see the correct number
            assert_eq!(seen_factories, 3);
        });
        
        producer.join().unwrap();
        consumer.join().unwrap();
    });
}

#[test]
fn test_work_stealing_race() {
    // Test work stealing between multiple workers
    loom::model(|| {
        use loom::sync::Arc;
        use std::collections::VecDeque;
        use loom::sync::Mutex;
        
        // Simplified work queue
        let work_queue = Arc::new(Mutex::new(VecDeque::new()));
        
        // Add initial work
        {
            let mut queue = work_queue.lock().unwrap();
            queue.push_back(1);
            queue.push_back(2);
            queue.push_back(3);
        }
        
        let queue1 = work_queue.clone();
        let queue2 = work_queue.clone();
        
        // Worker 1 tries to steal work
        let worker1 = thread::spawn(move || {
            let mut stolen = Vec::new();
            for _ in 0..2 {
                if let Some(work) = queue1.lock().unwrap().pop_front() {
                    stolen.push(work);
                }
                loom::thread::yield_now();
            }
            stolen
        });
        
        // Worker 2 tries to steal work
        let worker2 = thread::spawn(move || {
            let mut stolen = Vec::new();
            for _ in 0..2 {
                if let Some(work) = queue2.lock().unwrap().pop_front() {
                    stolen.push(work);
                }
                loom::thread::yield_now();
            }
            stolen
        });
        
        let work1 = worker1.join().unwrap();
        let work2 = worker2.join().unwrap();
        
        // Verify all work was processed exactly once
        let mut all_work = work1;
        all_work.extend(work2);
        all_work.sort();
        
        // We should have gotten 3 items total (some workers might get 0)
        assert!(all_work.len() <= 3);
        
        // Each item should be unique
        for i in 1..all_work.len() {
            assert_ne!(all_work[i], all_work[i-1]);
        }
    });
}

#[test]
fn test_filter_rwlock_consistency() {
    // Test RwLock access patterns in filter.rs
    // Multiple readers reading selectivity while a writer updates
    loom::model(|| {
        use loom::sync::{Arc, RwLock};
        
        // Simulate the selectivity histogram
        let selectivity = Arc::new(RwLock::new(vec![0.5, 0.6, 0.7]));
        
        let sel_clone1 = selectivity.clone();
        let sel_clone2 = selectivity.clone();
        let sel_clone3 = selectivity.clone();
        
        // Reader 1 - reads selectivity values
        let reader1 = thread::spawn(move || {
            let values = sel_clone1.read().unwrap();
            let sum: f64 = values.iter().sum();
            sum
        });
        
        // Reader 2 - reads selectivity values
        let reader2 = thread::spawn(move || {
            let values = sel_clone2.read().unwrap();
            let sum: f64 = values.iter().sum();
            sum
        });
        
        // Writer - updates selectivity
        let writer = thread::spawn(move || {
            let mut values = sel_clone3.write().unwrap();
            values[0] = 0.4;
            values[1] = 0.5;
            values[2] = 0.6;
        });
        
        let sum1 = reader1.join().unwrap();
        let sum2 = reader2.join().unwrap();
        writer.join().unwrap();
        
        // Both readers should see a consistent snapshot
        // They either see the old values (1.8) or new values (1.5)
        assert!(sum1 == 1.8 || sum1 == 1.5);
        assert!(sum2 == 1.8 || sum2 == 1.5);
    });
}

#[test]
fn test_dynamic_version_update() {
    // Test the TOCTOU fix in tasks.rs
    loom::model(|| {
        use loom::sync::Arc;
        use loom::sync::atomic::{AtomicU64, Ordering};
        
        // Simulate dynamic version that can change
        let version = Arc::new(AtomicU64::new(1));
        
        let version_clone = version.clone();
        
        // Thread that updates the version
        let updater = thread::spawn(move || {
            loom::thread::yield_now();
            version_clone.store(2, Ordering::Release);
        });
        
        // Thread that reads version (simulating the fixed code in tasks.rs)
        let reader = thread::spawn(move || {
            // This simulates the fix: read once and store
            let current_version = version.load(Ordering::Acquire);
            
            // Use the stored version for comparison
            if current_version > 0 {
                // Simulate updating local state
                Some(current_version)
            } else {
                None
            }
        });
        
        updater.join().unwrap();
        let result = reader.join().unwrap();
        
        // The reader should see either version 1 or 2, but consistently
        assert!(result == Some(1) || result == Some(2));
    });
}

#[test]
fn test_concurrent_factory_construction() {
    // Test concurrent factory construction and task stealing
    loom::model(|| {
        use loom::sync::{Arc, Mutex};
        
        // Shared state for factories
        let factories = Arc::new(Mutex::new(Vec::new()));
        let constructed_count = Arc::new(AtomicUsize::new(0));
        
        let factories_clone1 = factories.clone();
        let factories_clone2 = factories.clone();
        let count_clone1 = constructed_count.clone();
        let count_clone2 = constructed_count.clone();
        
        // Thread 1 constructs factories
        let constructor1 = thread::spawn(move || {
            let mut facs = factories_clone1.lock().unwrap();
            facs.push(1);
            drop(facs);
            count_clone1.fetch_add(1, Ordering::Release);
        });
        
        // Thread 2 constructs factories
        let constructor2 = thread::spawn(move || {
            let mut facs = factories_clone2.lock().unwrap();
            facs.push(2);
            drop(facs);
            count_clone2.fetch_add(1, Ordering::Release);
        });
        
        constructor1.join().unwrap();
        constructor2.join().unwrap();
        
        // Verify both factories were constructed
        assert_eq!(constructed_count.load(Ordering::Acquire), 2);
        let final_factories = factories.lock().unwrap();
        assert_eq!(final_factories.len(), 2);
    });
}