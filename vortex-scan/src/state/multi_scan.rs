// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Scan2;
use crate::state::pool::{ScanWorker, WorkerPool};
use parking_lot::Mutex;
use std::sync::Arc;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;

/// A way of orchestrating multiple scans across a worker pool.
#[derive(Clone)]
pub struct MultiScanPool {
    shared: Arc<Mutex<Shared>>,
}

impl MultiScanPool {
    pub fn new<I: IntoIterator<Item = VortexResult<Scan2>> + 'static>(pending: I) -> Self {
        let pending = Box::new(pending.into_iter());
        MultiScanPool {
            shared: Arc::new(Mutex::new(Shared {
                pending,
                pools: vec![],
            })),
        }
    }

    pub fn new_worker(&self) -> MultiScanWorker {
        let mut shared = self.shared.lock();

        let worker_idx = shared.pools.len();
        shared.pools.push(None);

        MultiScanWorker {
            worker_idx,
            shared: self.shared.clone(),
            current_worker: None,
        }
    }
}

struct Shared {
    /// A stream containing pending scans.
    pending: Box<dyn Iterator<Item = VortexResult<Scan2>>>,
    pools: Vec<Option<WorkerPool>>,
}

impl Shared {
    /// Attempt to create a new scan worker for the given worker index.
    fn next_worker(&mut self, worker_idx: usize) -> VortexResult<Option<ScanWorker>> {
        // First, we check that the worker's pool is indeed finished.
        if let Some(pool) = self.pools[worker_idx].take() {
            assert!(pool.is_finished(), "Worker pool is not finished")
        }

        // First, we attempt to pull a new worker from the pending stream.
        if let Some(scan) = self.pending.next() {
            // Create a new worker pool.
            let pool = scan?.into_worker_pool();
            self.pools[worker_idx] = Some(pool.clone());
            return Ok(Some(pool.new_worker()));
        }

        // If we have no more pending scans, then we try to create a new worker for another pool.
        for pool in &mut self.pools {
            if let Some(pool) = pool.as_ref() {
                return Ok(Some(pool.new_worker()));
            }
        }

        // Otherwise, we're done.
        Ok(None)
    }
}

pub struct MultiScanWorker {
    worker_idx: usize,
    shared: Arc<Mutex<Shared>>,
    current_worker: Option<ScanWorker>,
}

impl Iterator for MultiScanWorker {
    type Item = VortexResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if let Some(mut worker) = self.current_worker.take() {
                // Attempt to drive the current worker.
                if let Some(result) = worker.next() {
                    self.current_worker = Some(worker);
                    return Some(result);
                }
            }

            // Otherwise, if the worker is done, then we fetch a new one from the MultiScanPool.
            match self.shared.lock().next_worker(self.worker_idx) {
                Ok(Some(worker)) => {
                    self.current_worker = Some(worker);
                }
                Ok(None) => {
                    // There's no more work.
                    return None;
                }
                Err(e) => {
                    return Some(Err(e));
                }
            }
        }
    }
}
