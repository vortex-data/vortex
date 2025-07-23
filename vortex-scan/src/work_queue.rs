// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A work-stealing iterator that supports dynamically adding tasks from task factories.

use std::iter;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Relaxed, SeqCst};

use crossbeam_deque::{Steal, Stealer, Worker};
use crossbeam_queue::SegQueue;
use parking_lot::RwLock;
use vortex_error::VortexResult;

/// A factory that produces a vector of tasks.
pub type TaskFactory<T> = Box<dyn FnOnce() -> VortexResult<Vec<T>> + Send>;

/// A work-stealing queue that allows for dynamic task addition.
pub struct WorkStealingQueue<T> {
    state: Arc<State<T>>,
}

impl<T> Clone for WorkStealingQueue<T> {
    fn clone(&self) -> Self {
        Self {
            state: self.state.clone(),
        }
    }
}

impl<T> WorkStealingQueue<T> {
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

    pub fn new_iterator(self) -> WorkStealingIterator<T> {
        self.state.new_iterator()
    }
}

/// A work-stealing iterator that supports dynamically adding tasks from task factories.
///
/// Each task factory has affinity to a particular worker. After all factories have been
/// constructed, workers will attempt to steal tasks from each other until all tasks are processed.
///
/// Workers are constructed by cloning the iterator.
pub struct WorkStealingIterator<T> {
    state: Arc<State<T>>,
    worker: Worker<T>,
}

/// Shared state for the work queue.
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
    /// Create a new iterator.
    fn new_iterator(self: Arc<Self>) -> WorkStealingIterator<T> {
        let worker = Worker::new_fifo();

        // Register the new worker with the shared state.
        self.stealers.write().push(worker.stealer());

        WorkStealingIterator {
            state: self,
            worker,
        }
    }

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

impl<T> Clone for WorkStealingIterator<T> {
    fn clone(&self) -> Self {
        let worker = Worker::new_fifo();

        // Register the new worker with the shared state.
        self.state.stealers.write().push(worker.stealer());

        Self {
            state: self.state.clone(),
            worker,
        }
    }
}

impl<T> Iterator for WorkStealingIterator<T> {
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
