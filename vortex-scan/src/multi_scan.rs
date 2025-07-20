// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::iter;
use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::{Relaxed, SeqCst};

use crossbeam_deque::{Steal, Stealer, Worker};
use crossbeam_queue::SegQueue;
use futures::executor::LocalPool;
use futures::future::BoxFuture;
use parking_lot::RwLock;
use vortex_error::VortexResult;

use crate::ScanBuilder;

type ArrayFuture<T> = BoxFuture<'static, VortexResult<Option<T>>>;
type ScanBuilderFactory<T> = Box<dyn FnOnce() -> ScanBuilder<T> + Send + Sync>;

/// Coordinator to orchestrate multiple scan operations.
///
/// `MultiScan` allows to queue multiple scan operations in order to execute
/// them in parallel. In particular, this enables scanning multiple files.
pub struct MultiScan<T> {
    state: Arc<State<T>>,
}

struct State<T> {
    /// A queue of factories that lazily produce [`ScanBuilder`] instances.
    scan_builders: SegQueue<ScanBuilderFactory<T>>,

    /// The total number of scans that need to be constructed.
    num_scans: usize,
    /// How many scan builders have been constructed and had their tasks completely pushed
    /// into a worker queue.
    num_scans_constructed: AtomicUsize,

    /// The vector of stealers, one for each worker.
    stealers: RwLock<Vec<Stealer<ArrayFuture<T>>>>,
}

impl<T: 'static + Send + Sync> State<T> {
    /// Loads a scan and pushes its tasks into the given worker queue.
    ///
    /// Returns `true` if any tasks were pushed into the worker. Note that these tasks may have
    /// been stolen by the time the worker queue is checked.
    fn load_next_scan(&self, worker: &Worker<ArrayFuture<T>>) -> bool {
        if let Some(scan_builder_fn) = self.scan_builders.pop() {
            match scan_builder_fn().build() {
                Ok(tasks) => {
                    for task in tasks {
                        worker.push(Box::pin(task));
                    }
                    self.num_scans_constructed.fetch_add(1, SeqCst);
                }
                Err(err) => {
                    // If the scan builder fails, we can return an error.
                    worker.push(Box::pin(async { Err(err) }));
                }
            }
            true
        } else {
            false
        }
    }
}

impl<T> MultiScan<T> {
    /// Created with lazily constructed scan builders closures.
    pub fn new<I, F>(closures: I) -> Self
    where
        F: FnOnce() -> ScanBuilder<T> + 'static + Send + Sync,
        I: IntoIterator<Item = F>,
    {
        let scan_builders: SegQueue<ScanBuilderFactory<T>> = SegQueue::new();
        for closure in closures.into_iter() {
            scan_builders.push(Box::new(closure));
        }

        let num_scans = scan_builders.len();

        Self {
            state: Arc::new(State {
                scan_builders,
                num_scans_constructed: AtomicUsize::new(0),
                num_scans,
                stealers: RwLock::new(Vec::new()),
            }),
        }
    }

    /// Creates a new iterator to participate in the scan.
    ///
    /// The scan progresses when calling `next` on the iterator.
    pub fn new_scan_iterator(&self) -> MultiScanIterator<T> {
        let worker = Worker::new_fifo();

        // Register the worker with the shared state.
        self.state.stealers.write().push(worker.stealer());

        MultiScanIterator {
            state: self.state.clone(),
            worker,
            local_pool: LocalPool::new(),
        }
    }
}

/// Scan iterator to participate in a `MultiScan`.
pub struct MultiScanIterator<T> {
    state: Arc<State<T>>,

    worker: Worker<ArrayFuture<T>>,
    local_pool: LocalPool,
}

impl<T: Send + Sync + 'static> Iterator for MultiScanIterator<T> {
    type Item = VortexResult<T>;

    fn next(&mut self) -> Option<VortexResult<T>> {
        loop {
            // Try to consume tasks from our own worker queue.
            if let Some(task) = self.worker.pop() {
                return self.local_pool.run_until(task).transpose();
            }

            // Otherwise, try to load the next scan into our worker queue. We prefer to do this
            // before stealing from other workers so that each scan can be processed by a single
            // worker where possible for better locality.
            if self.state.load_next_scan(&self.worker) {
                // If we loaded a scan, continue to the next iteration and look for tasks again.
                continue;
            }

            // If we didn't load a scan, try to steal tasks from other workers.
            let did_steal = iter::repeat_with(|| {
                // This collect tries all stealers, exits early on the first successful steal,
                // or else tracks whether any steal requires a retry.
                self.state
                    .stealers
                    .read()
                    .iter()
                    .map(|stealer| stealer.steal_batch(&self.worker))
                    .collect::<Steal<()>>()
            })
            .find(|steal| !steal.is_retry())
            .and_then(|steal| steal.success())
            .is_some();

            if did_steal {
                // If we successfully stole some tasks, continue to the next iteration
                continue;
            } else {
                // Otherwise, if we have constructed all scans, _and_ there are no tasks
                // left to steal, then we can terminate.
                if self.state.num_scans_constructed.load(Relaxed) >= self.state.num_scans {
                    return None;
                } else {
                    // If there's more work to do, but no more tasks immediately available to
                    // steal, then we yield the thread to avoid a super hot loop. This only happens
                    // for the time it takes to invoke the final scan builder. If this becomes a
                    // problem, then we can use a mutex/condvar pair to notify workers when new
                    // tasks are available.
                    std::thread::yield_now();
                }
            }
        }
    }
}
