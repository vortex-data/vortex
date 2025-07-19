// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering::SeqCst;

use crossbeam_deque::{Steal, Stealer, Worker};
use crossbeam_queue::SegQueue;
use futures::executor::LocalPool;
use futures::future::BoxFuture;
use parking_lot::RwLock;
use vortex_error::VortexResult;

use crate::ScanBuilder;

type ArrayFuture<T> = BoxFuture<'static, VortexResult<Option<T>>>;
type ScanBuilderFactory<T> = Arc<SegQueue<Box<dyn FnOnce() -> ScanBuilder<T> + Send + Sync>>>;

/// Coordinator to orchestrate multiple scan operations.
///
/// `MultiScan` allows to queue multiple scan operations in order to execute
/// them in parallel. In particular, this enables scanning multiple files.
#[derive(Default)]
pub struct MultiScan<T> {
    scan_builder_factory: ScanBuilderFactory<T>,
    stealers: Arc<RwLock<Vec<Stealer<ArrayFuture<T>>>>>,
    next_stealer_id: Arc<AtomicUsize>,
}

impl<T> MultiScan<T> {
    pub fn new() -> Self {
        Self {
            scan_builder_factory: Arc::new(SegQueue::new()),
            stealers: Arc::new(RwLock::new(Vec::new())),
            next_stealer_id: Arc::new(AtomicUsize::new(0)),
        }
    }

    /// Add lazily constructed scan builders paired with their corresponding states.
    pub fn with_scan_builders<I, F>(self, closures: I) -> Self
    where
        F: FnOnce() -> ScanBuilder<T> + 'static + Send + Sync,
        I: IntoIterator<Item = F>,
    {
        for closure in closures.into_iter() {
            self.scan_builder_factory.push(Box::new(closure));
        }

        self
    }

    /// Creates a new iterator to participate in the scan.
    ///
    /// The scan progresses when calling `next` on the iterator.
    pub fn new_scan_iterator(&self) -> MultiScanIterator<T> {
        let worker = Worker::new_fifo();
        self.stealers.write().push(worker.stealer());

        MultiScanIterator {
            scan_builder_factory: self.scan_builder_factory.clone(),
            local_pool: LocalPool::new(),
            stealers: self.stealers.clone(),
            next_stealer_id: self.next_stealer_id.clone(),
            worker,
        }
    }
}

/// Scan iterator to participate in a `MultiScan`.
pub struct MultiScanIterator<T> {
    local_pool: LocalPool,
    worker: Worker<ArrayFuture<T>>,
    stealers: Arc<RwLock<Vec<Stealer<ArrayFuture<T>>>>>,
    next_stealer_id: Arc<AtomicUsize>,

    /// Thread-safe queue of closures that lazily produce [`ScanBuilder`] instances.
    /// This queue is shared across all iterators being created with `new_scan_iterator`.
    scan_builder_factory: ScanBuilderFactory<T>,
}

impl<T: Send + Sync + 'static> Iterator for MultiScanIterator<T> {
    type Item = VortexResult<T>;

    fn next(&mut self) -> Option<VortexResult<T>> {
        // Queue up tasks if the thread local queue is empty.
        if self.worker.is_empty() {
            if let Some(scan_builder_fn) = self.scan_builder_factory.pop() {
                match scan_builder_fn().build() {
                    Ok(tasks) => {
                        for task in tasks {
                            self.worker.push(Box::pin(task));
                        }
                    }
                    Err(err) => return Some(Err(err)),
                }
            } else {
                let stealer_count = self.stealers.read().len();
                let stealer_id = self.next_stealer_id.fetch_add(1, SeqCst) % stealer_count;

                for idx in 0..stealer_count {
                    // Round robin to ensure work is not always stolen from the same worker.
                    let stealer = &self.stealers.read()[(stealer_id + idx) % stealer_count];

                    // Attempt to steal ~half of the work and push it into `worker`.
                    if let Steal::Success(_) = stealer.steal_batch(&self.worker) {
                        break;
                    }
                }
            }
        }

        let task = self.worker.pop()?;

        self.local_pool.run_until(task).transpose()
    }
}
