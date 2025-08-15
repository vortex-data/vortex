// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::Scan2;
use crate::state::pool::{ScanWorker, WorkerPool};
use parking_lot::Mutex;
use std::sync::Arc;
use vortex_array::ArrayRef;
use vortex_array::iter::ArrayIterator;
use vortex_dtype::DType;
use vortex_error::{VortexResult, vortex_bail};

/// A way of orchestrating multiple scans across a worker pool.
#[derive(Clone)]
pub struct MultiScanPool {
    shared: Arc<Shared>,
}

pub type ScanFactory = Box<dyn FnOnce() -> VortexResult<Option<Scan2>> + Send>;

impl MultiScanPool {
    pub fn try_new<I: IntoIterator<Item = ScanFactory> + 'static>(pending: I) -> VortexResult<Self>
    where
        <I as IntoIterator>::IntoIter: Send,
    {
        let mut pending = pending.into_iter();

        // Find the first non-pruned scan from the stream.
        let first_scan = loop {
            let Some(factory) = pending.next() else {
                vortex_bail!("Must provide at least one scan")
            };
            let Some(scan) = factory()? else {
                pending.next();
                continue;
            };
            break scan;
        };

        let dtype = first_scan.dtype().clone();
        let pending = Box::new(pending);
        Ok(MultiScanPool {
            shared: Arc::new(Shared {
                dtype,
                state: Mutex::new(State {
                    first: Some(first_scan),
                    pending,
                    pools: vec![],
                }),
            }),
        })
    }

    pub fn new_worker(&self) -> MultiScanWorker {
        let mut shared = self.shared.state.lock();
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
    dtype: DType,
    state: Mutex<State>,
}

struct State {
    first: Option<Scan2>,
    pending: Box<dyn Iterator<Item = ScanFactory> + Send>,
    pools: Vec<Option<WorkerPool>>,
}

impl Shared {
    /// Attempt to create a new scan worker for the given worker index.
    ///
    /// Note(ngates): we intentionally do not hold on to the state mutex for very long.
    fn next_worker(&self, worker_idx: usize) -> VortexResult<Option<ScanWorker>> {
        loop {
            if let Some(pool) = self.next_worker_pool(worker_idx)? {
                return Ok(Some(pool.new_worker()));
            }

            // Otherwise, we're done.
            return Ok(None);
        }
    }

    fn next_worker_pool(&self, worker_idx: usize) -> VortexResult<Option<WorkerPool>> {
        loop {
            if let Some(factory) = self.next_scan_factory(worker_idx)? {
                let Some(scan) = factory()? else {
                    // Keep going until we find a non-pruned scan.
                    continue;
                };
                assert_eq!(scan.dtype(), &self.dtype, "Scans must have the same dtype");

                let pool = scan.into_worker_pool();
                self.state.lock().pools[worker_idx] = Some(pool.clone());
                return Ok(Some(pool));
            }

            // Otherwise, we try to create a new worker for another pool.
            for pool in &mut self.state.lock().pools {
                if let Some(pool) = pool.as_ref() {
                    return Ok(Some(pool.clone()));
                }
            }

            // If we reach here, it means there are no more scans available.
            return Ok(None);
        }
    }

    fn next_scan_factory(&self, worker_idx: usize) -> VortexResult<Option<ScanFactory>> {
        let mut state = self.state.lock();

        // First, we check that the worker's pool is indeed finished.
        if let Some(pool) = state.pools[worker_idx].take() {
            assert!(pool.is_finished(), "Worker pool is not finished")
        }

        // Check for the first scan.
        if let Some(scan) = state.first.take() {
            return Ok(Some(Box::new(move || Ok(Some(scan)))));
        }

        // Then, we attempt to pull a new worker from the pending stream.
        if let Some(factory) = state.pending.next() {
            return Ok(Some(factory));
        }

        Ok(None)
    }
}

pub struct MultiScanWorker {
    worker_idx: usize,
    shared: Arc<Shared>,
    current_worker: Option<ScanWorker>,
}

impl MultiScanWorker {
    pub fn idx(&self) -> usize {
        self.worker_idx
    }
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
            match self.shared.next_worker(self.worker_idx) {
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

impl ArrayIterator for MultiScanWorker {
    fn dtype(&self) -> &DType {
        &self.shared.dtype
    }
}
