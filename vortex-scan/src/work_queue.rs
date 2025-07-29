// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A work-stealing iterator that supports dynamically adding tasks from task factories.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Relaxed, SeqCst};
use std::{iter, thread};

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
                stealer_offset: Default::default(),
            }),
        }
    }

    pub fn new_iterator(self) -> WorkStealingIterator<T> {
        self.state.new_iterator()
    }
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

    /// An offset into the stealers vector, used to avoid skewed worker queues when stealing.
    stealer_offset: AtomicUsize,
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
        loop {
            if let Some(factory_fn) = self.task_factories.pop() {
                let tasks = factory_fn().inspect_err(|_| {
                    // In case of an error, increment the counter such that all other workers are able to terminate.
                    // `num_factories_constructed` is part of the loop condition when workers attempt to steal work.
                    self.num_factories_constructed.fetch_add(1, SeqCst);
                })?;
                let is_empty = tasks.is_empty();

                // Tasks *must* be pushed before `num_factories_constructed` is incremented.
                for task in tasks {
                    worker.push(task);
                }

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

    /// Reports whether there is any work left to steal.
    fn stealers_have_work(&self) -> bool {
        self.stealers
            .read()
            .iter()
            .any(|stealer| !stealer.is_empty())
    }

    /// Attempts to steal work from other workers, returns `true` if work was stolen.
    fn steal_work(&self, worker: &Worker<T>) -> Steal<()> {
        // Repeatedly attempt to steal work from other workers until there are no retries.
        iter::repeat_with(|| {
            // This collect tries all stealers, exits early on the first successful steal,
            // or else tracks whether any steal requires a retry.
            let guard = self.stealers.read();
            let num_stealers = guard.len();
            guard
                .iter()
                .cycle()
                .skip(self.stealer_offset.fetch_add(1, Relaxed) % num_stealers)
                .take(num_stealers)
                .map(|stealer| stealer.steal_batch(worker))
                .collect::<Steal<()>>()
        })
        .find(|steal| !steal.is_retry())
        .unwrap_or(Steal::Empty)
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

impl<T> Clone for WorkStealingIterator<T> {
    fn clone(&self) -> Self {
        self.state.clone().new_iterator()
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
                //
                // `steal_work` does have the side effect of stealing work, and we only want to loop
                // again if the result of an attempt of stealing results with `Retry`, for other cases
                // `Empty` and `Success` there is no point in trying again
                while self.state.num_factories_constructed.load(Relaxed) < self.state.num_factories
                    || self.state.stealers_have_work()
                {
                    if self.state.steal_work(&self.worker).is_success() {
                        break;
                    } else {
                        thread::yield_now();
                    }
                }
            }
        }

        // Attempt to pop a task from the worker queue.
        // Another worker may have stolen our tasks by this point. If that's the case, then we've
        // already finished loading the factories, and we're down to the last few tasks. Therefore,
        // it's ok for us to return `None` and terminate the iterator.
        Some(Ok(self.worker.pop()?))
    }
}
