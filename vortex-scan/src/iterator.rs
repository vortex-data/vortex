use core::panic;
use std::future::Future;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crossbeam_queue::SegQueue;
use dashmap::DashMap;
use futures::future::BoxFuture;
use futures::stream::{FuturesUnordered, StreamExt};
use vortex_array::ArrayRef;
use vortex_dtype::DType;
use vortex_error::VortexResult;
use vortex_expr::ExprRef;
use vortex_file::{VortexFile, VortexOpenOptions};

type ArrayFuture = BoxFuture<'static, VortexResult<Option<ArrayRef>>>;

pub struct MultiFileIterator {
    tokio_runtime: tokio::runtime::Runtime,
    file_paths: SegQueue<PathBuf>,
    files: SegQueue<VortexFile>,
    task_queues: Arc<DashMap<usize, SegQueue<(ArrayFuture, DType)>>>,
    projection_expr: Option<ExprRef>,
    filter_expr: Option<ExprRef>,
}

impl MultiFileIterator {
    pub fn new(num_threads: usize) -> Self {
        let thread_queues = Arc::new(DashMap::new());
        for thread_id in 0..num_threads {
            thread_queues.insert(thread_id, SegQueue::new());
        }

        Self {
            task_queues: thread_queues,
            file_paths: SegQueue::new(),
            files: SegQueue::new(),
            tokio_runtime: tokio::runtime::Runtime::new().unwrap(),
            projection_expr: None,
            filter_expr: None,
        }
    }

    pub fn with_file_paths<P, I>(self, file_paths: I) -> Self
    where
        P: AsRef<Path>,
        I: IntoIterator<Item = P>,
    {
        for path in file_paths.into_iter().map(|p| p.as_ref().to_path_buf()) {
            self.file_paths.push(path);
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

    pub fn push_scan_task<I, F>(&self, thread_id: usize, items: I, dtype: DType)
    where
        I: IntoIterator<Item = F>,
        F: Future<Output = VortexResult<Option<ArrayRef>>> + Send + 'static,
    {
        if let Some(task_queue) = self.task_queues.get(&thread_id) {
            for item in items {
                task_queue.push((Box::pin(item), dtype.clone()));
            }
        }
    }

    fn pop_scan_task(&self, preferred_thread: usize) -> Option<VortexResult<(ArrayFuture, DType)>> {
        if let Some(queue) = self.task_queues.get(&preferred_thread) {
            if let Some(array_future_tuple) = queue.pop() {
                return Some(Ok(array_future_tuple));
            }
        }
        None
    }

    fn push_scan_tasks_for_file(&self, file: VortexFile, thread_id: usize) -> VortexResult<()> {
        let scan_builder = file.scan()?;
        let scan_builder = scan_builder
            .with_executor(Arc::new(self.tokio_runtime.handle().clone()))
            .with_some_filter(self.filter_expr.clone())
            .with_projection(self.projection_expr.clone().unwrap());
        let (dtype, split_tasks) = scan_builder.build()?;

        // Get thread local task queue.
        let Some(task_queue) = self.task_queues.get(&thread_id) else {
            panic!("Thread local queue not found");
        };

        for task in split_tasks {
            task_queue.push((Box::pin(task), dtype.clone()));
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

        loop {
            let mut futures = FuturesUnordered::new();

            // Queue up tasks if the thread local queue is almost empty.
            if task_queue.len() <= 8 {
                if let Some(file) = self.files.pop() {
                    self.push_scan_tasks_for_file(file, thread_id).ok()?;
                } else if let Some(file_path) = self.file_paths.pop() {
                    let file = VortexOpenOptions::file().open_blocking(&file_path).ok()?;
                    self.push_scan_tasks_for_file(file, thread_id).ok()?;
                }
            }

            // TODO
            // - poll multiple futures concurrently instead of just one
            // - batch pop tasks with crossbeam deque, rather than looping
            if let Some(work_result) = self.pop_scan_task(thread_id) {
                match work_result {
                    Ok((future, _dtype)) => futures.push(future),
                    Err(e) => return Some(Err(e)),
                }
            }

            if futures.is_empty() {
                // Iterator is fully consumed.
                return None;
            }

            let result = self.tokio_runtime.block_on(async {
                while let Some(result) = futures.next().await {
                    match result {
                        Ok(Some(array)) => return Some(Ok(array)),
                        Ok(None) => continue, // No more data from this work item, try next
                        Err(e) => return Some(Err(e)),
                    }
                }
                None // All futures completed without yielding data
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
