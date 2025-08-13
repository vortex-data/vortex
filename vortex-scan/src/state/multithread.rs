// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! A scan driver for performing an ordered but multithreaded scan. This driver produces a single
//! blocking [`ArrayIterator`] that internally uses a thread pool to parallelize work.

use std::future::poll_fn;
use std::sync::LazyLock;
use std::task::Poll;

use futures::executor::{ThreadPool, ThreadPoolBuilder, block_on};
use futures::task::SpawnExt;
use vortex_array::ArrayRef;
use vortex_array::iter::ArrayIterator;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_err};

use crate::state::{Scan2, ScanTask, Scheduler, TaskSpawner};

static POOL: LazyLock<ThreadPool> = LazyLock::new(|| {
    ThreadPoolBuilder::new()
        .name_prefix("vortex-scan-multithread-")
        .create()
        .vortex_expect("Failed to create a thread pool for multithreaded scans")
});

impl Scan2 {
    pub fn into_iter_multithreaded(self) -> impl ArrayIterator {
        let dtype = self.ctx.dtype.clone();
        let scheduler = self.into_scheduler(Box::new(POOL.clone()));
        MultithreadedScan { dtype, scheduler }
    }
}

impl TaskSpawner for ThreadPool {
    fn spawn_task(&self, task: Box<dyn ScanTask>) {
        self.spawn(async move {
            // FIXME(ngates): this result will disappear.
            let _ = task.execute().vortex_expect("Failed to execute scan task");
        })
        .map_err(|e| vortex_err!("Failed to spawn task {e}"))
        .vortex_expect("Failed to spawn task onto thread pool")
    }
}

struct MultithreadedScan {
    dtype: DType,
    scheduler: Scheduler,
}

impl Iterator for MultithreadedScan {
    type Item = VortexResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.scheduler.finished {
                return None;
            }

            loop {
                match self.scheduler.make_progress() {
                    Poll::Ready(Ok(())) => continue,
                    Poll::Ready(Err(e)) => {
                        return Some(Err(e));
                    }
                    Poll::Pending => break,
                }
            }

            if let Some(result) = self.scheduler.output_buffer.pop_front() {
                return Some(result);
            }

            // Otherwise, we block waiting for a progress.
            if let Err(e) = block_on(poll_fn(|cx| self.scheduler.make_progress_with_cx(cx))) {
                return Some(Err(e));
            }
        }
    }
}

impl ArrayIterator for MultithreadedScan {
    fn dtype(&self) -> &DType {
        &self.dtype
    }
}
