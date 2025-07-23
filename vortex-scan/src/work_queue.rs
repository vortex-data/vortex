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
        if let Some(factory_fn) = self.task_factories.pop() {
            let tasks = factory_fn()?;
            for task in tasks {
                worker.push(task);
            }
            self.num_factories_constructed.fetch_add(1, SeqCst);
            Ok(true)
        } else {
            Ok(false)
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
    use rand::prelude::*;
    use std::collections::HashMap;
    use std::sync::atomic::Ordering::{Relaxed, SeqCst};
    use std::sync::atomic::{AtomicBool, AtomicUsize};
    use std::sync::{Arc, Barrier};
    use std::thread;
    use std::time::Duration;
    use vortex_error::vortex_bail;

    /// Fuzz test with varying numbers of workers and factories
    #[test]
    fn fuzz_worker_factory_combinations() {
        for _ in 0..100 {
            let mut rng = rand::rng();
            let num_factories = rng.random_range(1..=20);
            let num_workers = rng.random_range(1..=8);
            let tasks_per_factory = rng.random_range(0..=100);

            println!(
                "Testing: {} factories, {} workers, {} tasks/factory",
                num_factories, num_workers, tasks_per_factory
            );

            test_concurrent_processing(num_factories, num_workers, tasks_per_factory);
        }
    }

    /// Fuzz test with random factory execution times and failure rates
    #[test]
    fn fuzz_factory_reliability() {
        for iteration in 0..50 {
            let mut rng = rand::rng();
            let num_factories = rng.random_range(5..=15);
            let num_workers = rng.random_range(2..=6);
            let failure_rate = rng.random_range(0.0..=0.3); // 0-30% failure rate
            let max_delay_ms = rng.random_range(0..=10);

            println!(
                "Iteration {}: {} factories, {} workers, {:.1}% failure rate, {}ms max delay",
                iteration,
                num_factories,
                num_workers,
                failure_rate * 100.0,
                max_delay_ms
            );

            test_factory_failures_and_delays(
                num_factories,
                num_workers,
                failure_rate,
                max_delay_ms,
            );
        }
    }

    /// Fuzz test with workers joining and leaving at random times
    #[test]
    fn fuzz_dynamic_worker_lifecycle() {
        for _ in 0..30 {
            let mut rng = rand::rng();
            let num_factories = rng.random_range(10..=20);
            let initial_workers = rng.random_range(1..=4);
            let additional_workers = rng.random_range(0..=6);

            test_dynamic_workers(num_factories, initial_workers, additional_workers);
        }
    }

    /// Stress test with high contention scenarios
    #[test]
    fn fuzz_high_contention() {
        for _ in 0..20 {
            let mut rng = rand::rng();
            let scenario = rng.random_range(0..4);

            match scenario {
                0 => test_many_small_factories(rng.random_range(50..=200), rng.random_range(4..=8)),
                1 => test_few_large_factories(
                    rng.random_range(2..=5),
                    rng.random_range(2..=4),
                    rng.random_range(100..=500),
                ),
                2 => test_mixed_factory_sizes(rng.random_range(10..=30), rng.random_range(3..=6)),
                _ => test_empty_factories_mixed(rng.random_range(5..=20), rng.random_range(2..=4)),
            }
        }
    }

    /// Test edge cases with empty or single-task scenarios
    #[test]
    fn fuzz_edge_cases() {
        for _ in 0..50 {
            let mut rng = rand::rng();
            let scenario = rng.random_range(0..6);

            match scenario {
                0 => test_all_empty_factories(rng.random_range(1..=10), rng.random_range(1..=4)),
                1 => {
                    test_single_task_per_factory(rng.random_range(1..=20), rng.random_range(1..=8))
                }
                2 => test_single_factory_many_workers(rng.random_range(2..=8)),
                3 => test_many_factories_single_worker(rng.random_range(5..=50)),
                4 => test_interleaved_factory_loading(
                    rng.random_range(5..=15),
                    rng.random_range(2..=4),
                ),
                _ => test_rapid_worker_creation(rng.random_range(10..=20)),
            }
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
                vortex_bail!("Factory failed");
            }

            Ok((0..task_count).map(|i| (id, i)).collect())
        })
    }

    fn test_concurrent_processing(
        num_factories: usize,
        num_workers: usize,
        tasks_per_factory: usize,
    ) {
        let mut factories = Vec::new();
        for i in 0..num_factories {
            factories.push(create_factory(tasks_per_factory, i, 0, false));
        }

        let queue = WorkQueue::new(factories);
        let expected_total = num_factories * tasks_per_factory;

        // Spawn workers
        let barrier = Arc::new(Barrier::new(num_workers));
        let processed_count = Arc::new(AtomicUsize::new(0));
        let task_tracking = Arc::new(std::sync::Mutex::new(
            HashMap::<(usize, usize), usize>::new(),
        ));

        let handles: Vec<_> = (0..num_workers)
            .map(|_worker_id| {
                let queue = queue.clone();
                let barrier = barrier.clone();
                let processed_count = processed_count.clone();
                let task_tracking = task_tracking.clone();

                thread::spawn(move || {
                    barrier.wait();
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
                            Err(e) => panic!("Unexpected error: {:?}", e),
                        }
                    }

                    processed_count.fetch_add(local_count, SeqCst);
                    local_count
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

    fn test_factory_failures_and_delays(
        num_factories: usize,
        num_workers: usize,
        failure_rate: f64,
        max_delay_ms: u64,
    ) {
        let mut rng = rand::rng();
        let mut factories = Vec::new();
        let mut expected_tasks = 0;

        for i in 0..num_factories {
            let should_fail = rng.random::<f64>() < failure_rate;
            let delay = rng.random_range(0..=max_delay_ms);
            let task_count = if should_fail {
                0
            } else {
                rng.random_range(1..=50)
            };

            if !should_fail {
                expected_tasks += task_count;
            }

            factories.push(create_factory(task_count, i, delay, should_fail));
        }

        let queue = WorkQueue::new(factories);
        let processed_count = Arc::new(AtomicUsize::new(0));
        let error_count = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_workers)
            .map(|_| {
                let queue = queue.clone();
                let processed_count = processed_count.clone();
                let error_count = error_count.clone();

                thread::spawn(move || {
                    let mut iterator = queue.new_iterator();

                    while let Some(task_result) = iterator.next() {
                        match task_result {
                            Ok(_) => {
                                processed_count.fetch_add(1, SeqCst);
                            }
                            Err(_) => {
                                error_count.fetch_add(1, SeqCst);
                            }
                        }
                    }
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let total_processed = processed_count.load(SeqCst);
        let total_errors = error_count.load(SeqCst);

        println!(
            "✓ Processed: {}, Errors: {}, Expected: {}",
            total_processed, total_errors, expected_tasks
        );

        // We should process all successful tasks
        assert_eq!(total_processed, expected_tasks);
    }

    fn test_dynamic_workers(
        num_factories: usize,
        initial_workers: usize,
        additional_workers: usize,
    ) {
        let mut rng = rand::rng();
        let mut factories = Vec::new();
        let mut expected_total = 0;

        for i in 0..num_factories {
            let task_count = rng.random_range(5..=20);
            expected_total += task_count;
            factories.push(create_factory(task_count, i, 0, false));
        }

        let queue = WorkQueue::new(factories);
        let processed_count = Arc::new(AtomicUsize::new(0));
        let should_stop = Arc::new(AtomicBool::new(false));

        // Start initial workers
        let mut handles = Vec::new();
        for _ in 0..initial_workers {
            let queue = queue.clone();
            let processed_count = processed_count.clone();
            let should_stop = should_stop.clone();

            handles.push(thread::spawn(move || {
                let mut iterator = queue.new_iterator();
                let mut local_count = 0;

                while let Some(task_result) = iterator.next() {
                    if should_stop.load(Relaxed) {
                        break;
                    }

                    if let Ok(_) = task_result {
                        local_count += 1;
                    }
                }

                processed_count.fetch_add(local_count, SeqCst);
            }));
        }

        // Add additional workers with random delays
        for _ in 0..additional_workers {
            thread::sleep(Duration::from_millis(rng.random_range(1..=10)));

            let queue = queue.clone();
            let processed_count = processed_count.clone();
            let should_stop = should_stop.clone();

            handles.push(thread::spawn(move || {
                let mut iterator = queue.new_iterator();
                let mut local_count = 0;

                while let Some(task_result) = iterator.next() {
                    if should_stop.load(Relaxed) {
                        break;
                    }

                    if let Ok(_) = task_result {
                        local_count += 1;
                    }
                }

                processed_count.fetch_add(local_count, SeqCst);
            }));
        }

        // Wait for completion with timeout
        thread::sleep(Duration::from_millis(1000));
        should_stop.store(true, SeqCst);

        for handle in handles {
            handle.join().unwrap();
        }

        let total_processed = processed_count.load(SeqCst);
        assert_eq!(total_processed, expected_total);

        println!("✓ Dynamic workers processed {} tasks", total_processed);
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

    fn test_mixed_factory_sizes(num_factories: usize, num_workers: usize) {
        let mut rng = rand::rng();
        let mut factories = Vec::new();
        let mut expected_total = 0;

        for i in 0..num_factories {
            let size = *[1, 5, 10, 50, 100].choose(&mut rng).unwrap();
            expected_total += size;
            factories.push(create_factory(size, i, 0, false));
        }

        test_basic_processing(factories, num_workers, expected_total);
    }

    fn test_empty_factories_mixed(num_factories: usize, num_workers: usize) {
        let mut rng = rand::rng();
        let mut factories = Vec::new();
        let mut expected_total = 0;

        for i in 0..num_factories {
            let size = if rng.random::<f64>() < 0.3 {
                0
            } else {
                rng.random_range(1..=20)
            };
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

    fn test_interleaved_factory_loading(num_factories: usize, num_workers: usize) {
        // Create factories with small delays to interleave loading
        let factories: Vec<_> = (0..num_factories)
            .map(|i| create_factory(10, i, 1, false))
            .collect();

        test_basic_processing(factories, num_workers, num_factories * 10);
    }

    fn test_rapid_worker_creation(num_workers: usize) {
        let factories = vec![create_factory(50, 0, 0, false)];
        let queue = WorkQueue::new(factories);
        let processed_count = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_workers)
            .map(|_| {
                let queue = queue.clone();
                let processed_count = processed_count.clone();

                thread::spawn(move || {
                    let mut iterator = queue.new_iterator();
                    let mut local_count = 0;

                    while let Some(Ok(_)) = iterator.next() {
                        local_count += 1;
                    }

                    processed_count.fetch_add(local_count, SeqCst);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        assert_eq!(processed_count.load(SeqCst), 50);
    }

    fn test_basic_processing(
        factories: Vec<TaskFactory<(usize, usize)>>,
        num_workers: usize,
        expected_total: usize,
    ) {
        let queue = WorkQueue::new(factories);
        let processed_count = Arc::new(AtomicUsize::new(0));

        let handles: Vec<_> = (0..num_workers)
            .map(|_| {
                let queue = queue.clone();
                let processed_count = processed_count.clone();

                thread::spawn(move || {
                    let mut iterator = queue.new_iterator();
                    let mut local_count = 0;

                    while let Some(Ok(_)) = iterator.next() {
                        local_count += 1;
                    }

                    processed_count.fetch_add(local_count, SeqCst);
                })
            })
            .collect();

        for handle in handles {
            handle.join().unwrap();
        }

        let total_processed = processed_count.load(SeqCst);
        assert_eq!(total_processed, expected_total);

        println!("✓ Processed {} tasks as expected", total_processed);
    }
}
