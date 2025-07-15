// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use core::panic;
use std::future::Future;
use std::path::{Path, PathBuf};

use crossbeam_queue::SegQueue;
use dashmap::DashMap;
use futures::executor::LocalPool;
use futures::future::BoxFuture;
use futures::stream::{FuturesUnordered, StreamExt};
use vortex_array::ArrayRef;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_file::{VortexFile, VortexOpenOptions};

type ArrayFuture = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

pub struct MultiFileIterator {
    files: SegQueue<VortexFile>,
    filter_expr: Option<ExprRef>,
    local_pools: DashMap<usize, LocalPool>,
    paths: SegQueue<PathBuf>,
    polled_tasks: DashMap<usize, FuturesUnordered<ArrayFuture>>,
    projection_expr: Option<ExprRef>,
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
            paths: SegQueue::new(),
            files: SegQueue::new(),
            local_pools,
            projection_expr: None,
            filter_expr: None,
            polled_tasks: processed_tasks,
        }
    }

    pub fn with_file_paths<P, I>(self, file_paths: I) -> Self
    where
        P: AsRef<Path>,
        I: IntoIterator<Item = P>,
    {
        for path in file_paths.into_iter().map(|p| p.as_ref().to_path_buf()) {
            self.paths.push(path);
        }

        self
    }

    pub fn with_vortex_files<I>(self, vortex_files: I) -> Self
    where
        I: IntoIterator<Item = VortexFile>,
    {
        for vortex_file in vortex_files.into_iter() {
            self.files.push(vortex_file);
        }

        self
    }

    pub fn with_projection_expr(mut self, projection_expr: Option<ExprRef>) -> Self {
        self.projection_expr = projection_expr;
        self
    }

    pub fn with_filter_expr(mut self, filter_expr: Option<ExprRef>) -> Self {
        self.filter_expr = filter_expr;
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

    fn push_scan_tasks_for_file(&self, file: VortexFile, thread_id: usize) -> VortexResult<()> {
        let scan_builder = file
            .scan()?
            .with_some_filter(self.filter_expr.clone())
            .with_projection(self.projection_expr.clone().unwrap());

        let (_, split_tasks) = scan_builder.build()?;

        // Get thread local task queue.
        let Some(task_queue) = self.task_queues.get(&thread_id) else {
            panic!("Thread local queue not found");
        };

        for task in split_tasks {
            task_queue.push(Box::pin(task));
        }

        Ok(())
    }
}

impl MultiFileIterator {
    /// `next` is not implemented in terms of `impl Iterator` as `self`
    /// needs to be immutable in order to be shared across threads.
    pub fn next(&self, thread_id: usize) -> Option<VortexResult<ArrayRef>> {
        // Get thread local task queue.
        let Some(task_queue) = self.task_queues.get(&thread_id) else {
            panic!("Thread local queue not found");
        };

        let Some(mut processed_tasks) = self.polled_tasks.get_mut(&thread_id) else {
            panic!("Thread local processed tasks not found");
        };

        let Some(mut local_pool) = self.local_pools.get_mut(&thread_id) else {
            panic!("Thread local pool not found");
        };

        loop {
            // Queue up tasks if the thread local queue is almost empty.
            if task_queue.len() <= 4 {
                if let Some(file) = self.files.pop() {
                    self.push_scan_tasks_for_file(file, thread_id).ok()?;
                } else if let Some(file_path) = self.paths.pop() {
                    let file = VortexOpenOptions::file().open_blocking(&file_path).ok()?;
                    self.push_scan_tasks_for_file(file, thread_id).ok()?;
                }
                // TODO(Alex): worksteal tasks from other threads
            }

            // Poll 4 futures at a time.
            while processed_tasks.len() < 4 && !task_queue.is_empty() {
                if let Some(work_result) = self.pop_scan_task(thread_id) {
                    match work_result {
                        Ok(future) => processed_tasks.push(future),
                        Err(e) => return Some(Err(e)),
                    }
                }
            }

            if task_queue.is_empty() && processed_tasks.is_empty() {
                // All tasks have been fully processed.
                return None;
            }

            let result = local_pool.run_until(async {
                while let Some(result) = processed_tasks.next().await {
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
