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

    /// Attempts to steal work from other workers, returns `true` if work was stolen.
    fn steal_work(&self, worker: &Worker<ArrayFuture<T>>) -> Steal<()> {
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
        if self.worker.is_empty() && !self.state.load_next_scan(&self.worker) {
            // If there are no more scans to load, then there is at least one worker
            // constructing a scan and about to push some tasks.
            // We sit in a loop trying to steal some of those tasks, or else bail out when
            // all scans have been constructed, and we didn't manage to steal anything. To avoid
            // spinning too hot, we yield the thread each time we fail to steal work.
            while self.state.num_scans_constructed.load(Relaxed) < self.state.num_scans
                || !self.state.steal_work(&self.worker).is_empty()
            {
                if self.state.steal_work(&self.worker).is_success() {
                    break;
                } else {
                    std::thread::yield_now();
                }
            }
        }

        let task = self.worker.pop()?;
        self.local_pool.run_until(task).transpose()
    }
}
