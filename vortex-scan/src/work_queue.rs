// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Relaxed, SeqCst};

use crossbeam_deque::{Steal, Stealer, Worker};
use crossbeam_queue::SegQueue;
use parking_lot::RwLock;
use vortex_error::VortexResult;

/// A factory that produces a vector of tasks.
pub type TaskFactory<T> = Box<dyn FnOnce() -> VortexResult<Vec<T>> + Send + Sync>;

/// A work-stealing queue that supports dynamically adding tasks from task factories.
///
/// Each task factory has affinity to a particular worker. After all factories have been
/// constructed, workers will attempt to steal tasks from each other until all tasks are processed.
pub struct WorkQueue<T> {
    state: Arc<State<T>>,
}

impl<T> Clone for WorkQueue<T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

struct State<T> {
    /// A queue of factories that lazily produce tasks of type `T`.
    task_factories: SegQueue<TaskFactory<T>>,

    /// The total number of task factories that need to be constructed.
    num_factories: usize,

    /// How many factories have been constructed and had their tasks completely pushed into
    /// a worker queue.
    num_factories_constructed: AtomicUsize,

    /// The vector of stealers, one for each worker.
    stealers: RwLock<Vec<Stealer<T>>>,
}

impl<T> State<T> {
    /// Loads a factory and pushes its tasks into the given worker queue.
    ///
    /// Returns `true` if any tasks were pushed into the worker. Note that these tasks may have
    /// been stolen by the time the worker queue is checked.
    fn load_next_factory(&self, worker: &Worker<T>) -> VortexResult<bool> {
        loop {
            if let Some(factory_fn) = self.task_factories.pop() {
                let tasks = factory_fn()?;
                let is_empty = tasks.is_empty();
                for task in tasks {
                    worker.push(task);
                }

                // We **MUST** push the tasks into the worker before incrementing this counter.
                self.num_factories_constructed.fetch_add(1, SeqCst);

                // Keep looping until we find a factory that has pushed tasks.
                if !is_empty {
                    return Ok(true);
                }
            } else {
                return Ok(false);
            }
        }
    }

    /// Attempts to steal work from other workers, returns `true` if work was stolen.
    fn steal_work(&self, worker: &Worker<T>) -> Steal<()> {
        // Repeatedly attempt to steal work from other workers until there are no retries.
        iter::repeat_with(|| {
            // This collect tries all stealers, exits early on the first successful steal,
            // or else tracks whether any steal requires a retry.
            self.stealers
                .read()
                .iter()
                .map(|stealer| stealer.steal_batch(worker))
                .collect::<Steal<()>>()
        })
        .find(|steal| !steal.is_retry())
        .unwrap_or(Steal::Empty)
    }
}

impl<T> WorkQueue<T> {
    /// Created with lazily constructed task factories.
    pub fn new<I>(factories: I) -> Self
    where
        I: IntoIterator<Item = TaskFactory<T>>,
    {
        let queue: SegQueue<TaskFactory<T>> = SegQueue::new();
        for factory in factories.into_iter() {
            queue.push(factory);
        }
        let num_factories = queue.len();

        Self {
            state: Arc::new(State {
                task_factories: queue,
                num_factories_constructed: AtomicUsize::new(0),
                num_factories,
                stealers: RwLock::new(Vec::new()),
            }),
        }
    }

    /// Creates a new worker to participate.
    ///
    /// The scan progresses when calling `next` on the iterator.
    pub fn new_iterator(&self) -> WorkQueueIterator<T> {
        let worker = Worker::new_fifo();

        // Register the worker with the shared state.
        self.state.stealers.write().push(worker.stealer());

        WorkQueueIterator {
            state: self.state.clone(),
            worker,
        }
    }
}

/// Iterator yield tasks from the work-stealing queue.
pub struct WorkQueueIterator<T> {
    state: Arc<State<T>>,
    worker: Worker<T>,
}

impl<T> Iterator for WorkQueueIterator<T> {
    type Item = VortexResult<T>;

    fn next(&mut self) -> Option<VortexResult<T>> {
        if self.worker.is_empty() {
            let next_factory_loaded = match self.state.load_next_factory(&self.worker) {
                Ok(next_factory_loaded) => next_factory_loaded,
                Err(e) => return Some(Err(e)),
            };

            if !next_factory_loaded {
                // If there are no more factories to load, then there is at least one worker
                // constructing a factory and about to push some tasks.
                //
                // We sit in a loop trying to steal some of those tasks, or else bail out when
                // all scans have been constructed, and we didn't manage to steal anything. To avoid
                // spinning too hot, we yield the thread each time we fail to steal work.
                while self.state.num_factories_constructed.load(Relaxed) < self.state.num_factories
                    || !self.state.steal_work(&self.worker).is_empty()
                {
                    if self.state.steal_work(&self.worker).is_success() {
                        break;
                    } else {
                        std::thread::yield_now();
                    }
                }
            }
        }

        Some(Ok(self.worker.pop()?))
    }
}

#[cfg(test)]
mod fuzz_tests {
    use super::*;
    use itertools::Itertools;
    use rand::prelude::*;
    use std::collections::HashMap;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering::SeqCst;
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;
    use vortex_error::vortex_bail;

    #[test]
    fn test_worker_factory_combinations() {
        let factory_counts = [1, 2, 5, 10, 20, 50];
        let worker_counts = [1, 2, 3, 4, 8];
        let task_counts = [0, 1, 5, 10, 25, 100];

        for &num_factories in &factory_counts {
            for &num_workers in &worker_counts {
                for &tasks_per_factory in &task_counts {
                    println!(
                        "Testing: {} factories, {} workers, {} tasks/factory",
                        num_factories, num_workers, tasks_per_factory
                    );

                    test_concurrent_processing(num_factories, num_workers, tasks_per_factory);
                }
            }
        }
    }

    #[test]
    fn test_dynamic_worker_patterns() {
        let scenarios = [(10, 1, 2), (15, 2, 2), (12, 2, 1)];

        for &(num_factories, initial_workers, additional_workers) in &scenarios {
            println!(
                "Testing dynamic workers: {} factories, {} initial, {} additional",
                num_factories, initial_workers, additional_workers
            );
            test_dynamic_workers(num_factories, initial_workers, additional_workers);
        }
    }

    #[test]
    fn test_contention_scenarios() {
        // Many small factories
        let small_factory_tests = [(50, 4), (100, 6), (200, 8)];
        for &(factories, workers) in &small_factory_tests {
            println!(
                "Testing many small factories: {} factories, {} workers",
                factories, workers
            );
            test_many_small_factories(factories, workers);
        }

        // Few large factories
        let large_factory_tests = [(2, 4, 500), (5, 6, 200), (3, 8, 1000)];
        for &(factories, workers, tasks) in &large_factory_tests {
            println!(
                "Testing few large factories: {} factories, {} workers, {} tasks each",
                factories, workers, tasks
            );
            test_few_large_factories(factories, workers, tasks);
        }

        // Mixed sizes with deterministic patterns
        let mixed_tests = [(10, 4), (20, 6), (30, 8)];
        for &(factories, workers) in &mixed_tests {
            println!(
                "Testing mixed factory sizes: {} factories, {} workers",
                factories, workers
            );
            test_mixed_factory_sizes_deterministic(factories, workers);
        }

        // Empty factories mixed in
        let empty_mixed_tests = [(10, 3), (20, 4), (15, 6)];
        for &(factories, workers) in &empty_mixed_tests {
            println!(
                "Testing with empty factories: {} factories, {} workers",
                factories, workers
            );
            test_empty_factories_mixed_deterministic(factories, workers);
        }
    }

    /// Test edge cases systematically
    #[test]
    fn test_edge_cases() {
        // All empty factories
        let empty_tests = [(5, 2), (10, 4), (20, 6)];
        for &(factories, workers) in &empty_tests {
            println!(
                "Testing all empty factories: {} factories, {} workers",
                factories, workers
            );
            test_all_empty_factories(factories, workers);
        }

        // Single task per factory
        let single_tests = [(10, 3), (20, 5), (50, 8)];
        for &(factories, workers) in &single_tests {
            println!(
                "Testing single task per factory: {} factories, {} workers",
                factories, workers
            );
            test_single_task_per_factory(factories, workers);
        }

        // Single factory, many workers
        let single_factory_tests = [2, 4, 6, 8, 12];
        for &workers in &single_factory_tests {
            println!("Testing single factory, {} workers", workers);
            test_single_factory_many_workers(workers);
        }

        // Many factories, single worker
        let many_factory_tests = [5, 10, 25, 50, 100];
        for &factories in &many_factory_tests {
            println!("Testing {} factories, single worker", factories);
            test_many_factories_single_worker(factories);
        }
    }

    // Helper function to create a factory that produces a specific number of tasks
    fn create_factory(
        task_count: usize,
        id: usize,
        delay_ms: u64,
        should_fail: bool,
    ) -> TaskFactory<(usize, usize)> {
        Box::new(move || {
            if delay_ms > 0 {
                thread::sleep(Duration::from_millis(delay_ms));
            }

            if should_fail {
                vortex_bail!("Factory {} failed", id);
            }

            Ok((0..task_count).map(|i| (id, i)).collect())
        })
    }

    fn test_concurrent_processing(
        num_factories: usize,
        num_workers: usize,
        tasks_per_factory: usize,
    ) {
        let factories: Vec<_> = (0..num_factories)
            .map(|i| create_factory(tasks_per_factory, i, 0, false))
            .collect();

        let queue = WorkQueue::new(factories);
        let expected_total = num_factories * tasks_per_factory;

        // Spawn workers with barrier for synchronized start
        let barrier = Arc::new(Barrier::new(num_workers));
        let processed_count = Arc::new(AtomicUsize::new(0));
        let task_tracking = Arc::new(std::sync::Mutex::new(
            HashMap::<(usize, usize), usize>::new(),
        ));

        let handles: Vec<_> = (0..num_workers)
            .map(|worker_id| {
                let queue = queue.clone();
                let barrier = barrier.clone();
                let processed_count = processed_count.clone();
                let task_tracking = task_tracking.clone();

                thread::spawn(move || {
                    barrier.wait(); // Synchronized start
                    let mut iterator = queue.new_iterator();
                    let mut local_count = 0;

                    while let Some(task_result) = iterator.next() {
                        match task_result {
                            Ok(task) => {
                                // Track this task to detect duplicates
                                {
                                    let mut tracking = task_tracking.lock().unwrap();
                                    *tracking.entry(task).or_insert(0) += 1;
                                }
                                local_count += 1;
                            }
                            Err(e) => panic!("Worker {} unexpected error: {:?}", worker_id, e),
                        }
                    }

                    processed_count.fetch_add(local_count, SeqCst);
                    (worker_id, local_count)
                })
            })
            .collect();

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let total_processed = processed_count.load(SeqCst);

        // Verify results
        assert_eq!(
            total_processed, expected_total,
            "Expected {} tasks, but processed {}",
            expected_total, total_processed
        );

        // Check for duplicate processing
        let tracking = task_tracking.lock().unwrap();
        for (task, count) in tracking.iter() {
            assert_eq!(*count, 1, "Task {:?} was processed {} times", task, count);
        }

        println!(
            "✓ Workers processed: {:?}, Total: {}",
            results, total_processed
        );
    }

    fn test_dynamic_workers(
        num_factories: usize,
        initial_workers: usize,
        additional_workers: usize,
    ) {
        let mut rng = StdRng::seed_from_u64(num_factories as u64 * 100 + initial_workers as u64);
        let mut factories = Vec::new();
        let mut expected_total = 0;

        for i in 0..num_factories {
            let task_count = rng.random_range(5..=15); // Reduced from 5..=20
            expected_total += task_count;
            factories.push(create_factory(task_count, i, 0, false));
        }

        let queue = WorkQueue::new(factories);
        let processed_count = Arc::new(AtomicUsize::new(0));
        let start_barrier = Arc::new(Barrier::new(initial_workers));

        // Start initial workers
        let mut handles = Vec::new();
        for worker_id in 0..initial_workers {
            let queue = queue.clone();
            let processed_count = processed_count.clone();
            let start_barrier = start_barrier.clone();

            handles.push(thread::spawn(move || {
                start_barrier.wait();
                let mut iterator = queue.new_iterator();
                let mut local_count = 0;

                while let Some(task_result) = iterator.next() {
                    if let Ok(_) = task_result {
                        local_count += 1;
                    }
                }

                processed_count.fetch_add(local_count, SeqCst);
                (worker_id, local_count)
            }));
        }

        // Add additional workers with deterministic delays
        for worker_id in initial_workers..(initial_workers + additional_workers) {
            thread::sleep(Duration::from_millis(2)); // Reduced from 5ms

            let queue = queue.clone();
            let processed_count = processed_count.clone();

            handles.push(thread::spawn(move || {
                let mut iterator = queue.new_iterator();
                let mut local_count = 0;

                while let Some(task_result) = iterator.next() {
                    if let Ok(_) = task_result {
                        local_count += 1;
                    }
                }

                processed_count.fetch_add(local_count, SeqCst);
                (worker_id, local_count)
            }));
        }

        let results: Vec<_> = handles.into_iter().map(|h| h.join().unwrap()).collect();
        let total_processed = processed_count.load(SeqCst);

        assert_eq!(total_processed, expected_total);
        println!(
            "✓ Dynamic workers {:?} processed {} tasks",
            results, total_processed
        );
    }

    fn test_many_small_factories(num_factories: usize, num_workers: usize) {
        let factories: Vec<_> = (0..num_factories)
            .map(|i| create_factory(1, i, 0, false))
            .collect();

        test_basic_processing(factories, num_workers, num_factories);
    }

    fn test_few_large_factories(
        num_factories: usize,
        num_workers: usize,
        tasks_per_factory: usize,
    ) {
        let factories: Vec<_> = (0..num_factories)
            .map(|i| create_factory(tasks_per_factory, i, 0, false))
            .collect();

        test_basic_processing(factories, num_workers, num_factories * tasks_per_factory);
    }

    fn test_mixed_factory_sizes_deterministic(num_factories: usize, num_workers: usize) {
        let sizes = [1, 5, 10, 25, 50, 100];
        let mut factories = Vec::new();
        let mut expected_total = 0;

        for i in 0..num_factories {
            let size = sizes[i % sizes.len()];
            expected_total += size;
            factories.push(create_factory(size, i, 0, false));
        }

        test_basic_processing(factories, num_workers, expected_total);
    }

    fn test_empty_factories_mixed_deterministic(num_factories: usize, num_workers: usize) {
        let mut factories = Vec::new();
        let mut expected_total = 0;

        for i in 0..num_factories {
            // Every 3rd factory is empty
            let size = if i % 3 == 0 { 0 } else { 10 };
            expected_total += size;
            factories.push(create_factory(size, i, 0, false));
        }

        test_basic_processing(factories, num_workers, expected_total);
    }

    fn test_all_empty_factories(num_factories: usize, num_workers: usize) {
        let factories: Vec<_> = (0..num_factories)
            .map(|i| create_factory(0, i, 0, false))
            .collect();

        test_basic_processing(factories, num_workers, 0);
    }

    fn test_single_task_per_factory(num_factories: usize, num_workers: usize) {
        let factories: Vec<_> = (0..num_factories)
            .map(|i| create_factory(1, i, 0, false))
            .collect();

        test_basic_processing(factories, num_workers, num_factories);
    }

    fn test_single_factory_many_workers(num_workers: usize) {
        let factories = vec![create_factory(100, 0, 0, false)];
        test_basic_processing(factories, num_workers, 100);
    }

    fn test_many_factories_single_worker(num_factories: usize) {
        let factories: Vec<_> = (0..num_factories)
            .map(|i| create_factory(5, i, 0, false))
            .collect();

        test_basic_processing(factories, 1, num_factories * 5);
    }

    fn test_basic_processing(
        factories: Vec<TaskFactory<(usize, usize)>>,
        num_workers: usize,
        expected_total: usize,
    ) {
        let queue = WorkQueue::new(factories);
        let processed_count = Arc::new(AtomicUsize::new(0));

        let results: Vec<_> = (0..num_workers)
            .map(|worker_id| {
                let queue = queue.clone();
                let processed_count = processed_count.clone();

                thread::spawn(move || {
                    let mut iterator = queue.new_iterator();
                    let mut local_count = 0;

                    while let Some(Ok(_)) = iterator.next() {
                        local_count += 1;
                    }

                    processed_count.fetch_add(local_count, SeqCst);
                    (worker_id, local_count)
                })
            })
            .map(|h| h.join())
            .try_collect()
            .unwrap();

        let total_processed = processed_count.load(SeqCst);

        assert_eq!(total_processed, expected_total);
        println!(
            "✓ Workers {:?} processed {} tasks as expected",
            results, total_processed
        );
    }
}
