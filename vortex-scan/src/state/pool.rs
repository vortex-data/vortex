// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! This module contains a scan driver for integrating with query engines that use a blocking
//! multithreaded thread model.
//!
//! The scan is wrapped in a `WorkerPool` from which any number of worker iterators can be spawned.
//! Each iterator will perform blocking CPU-intensive work as well as driving the scan's scheduler
//! and I/O loop.
//!
//! NOTE: the results emitted from a worker pool scan have no ordering guarantees.

use std::future::poll_fn;
use std::iter;
use std::sync::Arc;
use std::task::Poll;

use crossbeam_deque::{Injector, Stealer, Worker};
use futures::executor::block_on;
use parking_lot::{Mutex, RwLock};
use vortex_array::ArrayRef;
use vortex_array::iter::ArrayIterator;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::state::{Scan2, ScanTask, Scheduler, TaskSpawner};

impl Scan2 {
    pub fn into_worker_pool(self) -> WorkerPool {
        // We instruct the scheduler to spawn tasks via the injector.
        // NOTE(ngates): in the future, the scheduler should spawn tasks with a worker affinity,
        //  and that can be used to inject tasks into a specific worker.
        let injector = Arc::new(Injector::new());
        let task_spawner: Box<dyn TaskSpawner> = Box::new(injector.clone());

        let shared = Arc::new(Shared {
            dtype: self.ctx.dtype.clone(),
            scheduler: Mutex::new(self.into_scheduler(task_spawner)),
            injector,
            stealers: Default::default(),
        });

        WorkerPool { shared }
    }
}

impl TaskSpawner for Arc<Injector<Box<dyn ScanTask>>> {
    fn spawn_task(&self, task: Box<dyn ScanTask>) {
        self.push(task);
    }
}

pub struct WorkerPool {
    shared: Arc<Shared>,
}

struct Shared {
    /// The DType of the projection
    dtype: DType,
    scheduler: Mutex<Scheduler>,
    injector: Arc<Injector<Box<dyn ScanTask>>>,
    stealers: RwLock<Vec<Stealer<Box<dyn ScanTask>>>>,
}

impl WorkerPool {
    /// Create a new worker for the scan.
    pub fn new_worker(&self) -> ScanWorker {
        let worker = Worker::new_fifo();
        self.shared.stealers.write().push(worker.stealer());

        ScanWorker {
            shared: self.shared.clone(),
            worker,
        }
    }
}

pub struct ScanWorker {
    shared: Arc<Shared>,
    worker: Worker<Box<dyn ScanTask>>,
}

impl Iterator for ScanWorker {
    type Item = VortexResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // We try to take the scheduler lock to drive it.
            if let Some(mut scheduler) = self.shared.scheduler.try_lock() {
                // First, we drain the output buffer since a worker scan returns results directly
                // from the scan tasks.
                while let Some(output) = scheduler.output_buffer.pop_front() {
                    if let Err(e) = output {
                        return Some(Err(e));
                    }
                }

                // If the scheduler is finished, we return None since we know all CPU tasks have
                // been completed.
                if scheduler.finished {
                    assert!(self.worker.is_empty());
                    return None;
                }

                // Otherwise, we drive the scheduler.
                loop {
                    match scheduler.make_progress() {
                        Poll::Ready(Ok(())) => {
                            continue;
                        }
                        Poll::Ready(Err(e)) => {
                            return Some(Err(e));
                        }
                        Poll::Pending => {
                            break;
                        }
                    };
                }
            }

            // Once we've performed our civic duty of driving the scheduler, we check if we have
            // any tasks of our own to execute.
            if let Some(task) = self.find_task() {
                if let Some(array) = task.execute() {
                    // We can immediately return the Array result since we don't care about
                    // ordering.
                    return Some(Ok(array));
                }

                // If the task did not produce a result (e.g. a filter conjunct), then we continue
                // to the beginning of the loop and attempt to drive the scheduler again.
                continue;
            }

            // If we've exhausted our own tasks, then we force ourselves to acquire the scheduler
            // lock.
            // TODO(ngates): this only works if the mutex is "fair", in other words, that all
            //  threads waiting on the same lock eventually get released. We should experiment
            //  with Parking Lot's default mutex which eventually falls back to fair scheduling,
            //  and if not, use [`parking_lot::FairMutex`] instead.
            let mut scheduler = self.shared.scheduler.lock();
            if scheduler.finished {
                assert!(self.worker.is_empty());
                return None;
            }

            // Otherwise, we sit waiting for an event to unblock the scheduler. This avoids us
            // sitting in a busy loop.
            if let Err(e) = block_on(poll_fn(|cx| scheduler.make_progress_with_cx(cx))) {
                return Some(Err(e));
            }
        }
    }
}

impl ScanWorker {
    /// Find the next CPU task.
    fn find_task(&mut self) -> Option<Box<dyn ScanTask>> {
        // Pop a task from the local queue, if not empty.
        self.worker.pop().or_else(|| {
            // Otherwise, we need to look for a task elsewhere.
            iter::repeat_with(|| {
                // Try stealing a batch of tasks from the global queue.
                self.shared
                    .injector
                    .steal_batch_and_pop(&self.worker)
                    // Or try stealing a task from one of the other threads.
                    .or_else(|| {
                        self.shared
                            .stealers
                            .read()
                            .iter()
                            .map(|s| s.steal())
                            .collect()
                    })
            })
            // Loop while no task was stolen and any steal operation needs to be retried.
            .find(|s| !s.is_retry())
            // Extract the stolen task if there is one.
            .and_then(|s| s.success())
        })
    }
}

impl ArrayIterator for ScanWorker {
    fn dtype(&self) -> &DType {
        &self.shared.dtype
    }
}

impl Drop for ScanWorker {
    fn drop(&mut self) {
        // Drain our local queue into the injector.
        // It's fiddly to remove the stealer, but that's ok, because there's no tasks to steal
        // anyway. It's just dead weight.
        while let Some(task) = self.worker.pop() {
            self.shared.injector.push(task);
        }
    }
}
