// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::panic;

use crossbeam_queue::SegQueue;
use dashmap::DashMap;
use futures::executor::LocalPool;
use futures::future::BoxFuture;
use futures::stream::{FuturesUnordered, StreamExt};
use vortex_array::ArrayRef;
use vortex_error::VortexResult;

use crate::ScanBuilder;

type ArrayFuture = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

pub struct MultiFileIterator {
    local_pools: DashMap<usize, LocalPool>,
    scan_builder_fns: SegQueue<Box<dyn FnOnce() -> ScanBuilder<ArrayRef>>>,
    polled_tasks: DashMap<usize, FuturesUnordered<ArrayFuture>>,
    task_queues: DashMap<usize, SegQueue<ArrayFuture>>,
}

impl MultiFileIterator {
    pub fn new(num_threads: usize) -> Self {
        let thread_queues = DashMap::new();
        let processed_tasks = DashMap::new();
        let local_pools = DashMap::new();
        for thread_id in 0..num_threads {
            thread_queues.insert(thread_id, SegQueue::new());
            processed_tasks.insert(thread_id, FuturesUnordered::new());
            local_pools.insert(thread_id, LocalPool::new());
        }

        Self {
            task_queues: thread_queues,
            scan_builder_fns: SegQueue::new(),
            local_pools,
            polled_tasks: processed_tasks,
        }
    }

    pub fn with_scan_builders<I, F>(self, closures: I) -> Self
    where
        F: FnOnce() -> ScanBuilder<ArrayRef> + 'static,
        I: IntoIterator<Item = F>,
    {
        for closure in closures.into_iter() {
            self.scan_builder_fns.push(Box::new(closure));
        }

        self
    }

    fn pop_scan_task(&self, preferred_thread: usize) -> Option<VortexResult<ArrayFuture>> {
        if let Some(queue) = self.task_queues.get(&preferred_thread) {
            if let Some(array_future_tuple) = queue.pop() {
                return Some(Ok(array_future_tuple));
            }
        }
        None
    }
}

impl MultiFileIterator {
    /// `next` is not implemented in terms of `impl Iterator` as `self`
    /// needs to be immutable in order to be shared across threads.
    pub fn next(&self, thread_id: usize) -> Option<VortexResult<ArrayRef>> {
        let Some(task_queue) = self.task_queues.get(&thread_id) else {
            panic!("Thread local queue not found");
        };

        let Some(mut polled_tasks) = self.polled_tasks.get_mut(&thread_id) else {
            panic!("Thread local processed tasks not found");
        };

        let Some(mut local_pool) = self.local_pools.get_mut(&thread_id) else {
            panic!("Thread local pool not found");
        };

        loop {
            // Queue up tasks if the thread local queue is almost empty.
            if task_queue.len() <= 4 {
                if let Some(scan_builder_fn) = self.scan_builder_fns.pop() {
                    let split_tasks = scan_builder_fn().build().ok()?.1;
                    for task in split_tasks {
                        task_queue.push(Box::pin(task));
                    }
                }
                // TODO(Alex): worksteal tasks from other threads
            }

            // Poll one future at a time. Polling multiple futures at
            // the same time leads to contention within a layout reader.
            if let Some(work_result) = self.pop_scan_task(thread_id) {
                match work_result {
                    Ok(future) => polled_tasks.push(future),
                    Err(e) => return Some(Err(e)),
                }
            }

            if task_queue.is_empty() && polled_tasks.is_empty() {
                // All tasks have been fully processed.
                return None;
            }

            let result = local_pool.run_until(async {
                while let Some(result) = polled_tasks.next().await {
                    match result {
                        Ok(Some(array)) => return Some(Ok(array)),
                        Ok(None) => continue,
                        Err(e) => return Some(Err(e)),
                    }
                }
                None
            });

            match result {
                Some(Ok(array)) => return Some(Ok(array)),
                Some(Err(e)) => return Some(Err(e)),
                None => continue, // Try next batch of futures
            }
        }
    }
}

unsafe impl Send for MultiFileIterator {}
unsafe impl Sync for MultiFileIterator {}
