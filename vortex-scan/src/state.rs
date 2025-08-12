// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::filter::FilterExpr;
use crate::tasks::TaskContext;
use crossbeam_deque::{Injector, Stealer, Worker};
use dashmap::DashMap;
use futures::executor::block_on;
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use futures::{FutureExt, StreamExt};
use parking_lot::{Mutex, RwLock};
use std::collections::VecDeque;
use std::iter;
use std::ops::Range;
use std::sync::{Arc, LazyLock};
use std::task::{Context, Poll, Wake, Waker};
use vortex_array::ArrayRef;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexResult, vortex_err};
use vortex_layout::segments::{SegmentId, SegmentSource};
use vortex_layout::{ArrayEvaluation, MaskEvaluation};
use vortex_utils::aliases::hash_set::HashSet;

pub struct Scan2 {
    scheduler: Mutex<Scheduler>,
}

impl Scan2 {
    pub fn try_new(ranges: Vec<Range<u64>>, ctx: TaskContext<ArrayRef>) -> VortexResult<Self> {
        todo!()
    }
}

struct GlobalState {
    scheduler: Mutex<Scheduler>,

    /// Information about each split in the scan.
    splits: Vec<Split>,

    /// The working set of segments.
    segments: Arc<DashMap<SegmentId, ByteBuffer>>,
    /// The filter expression for the scan.
    filter_expr: Option<Arc<FilterExpr>>,

    injector: Injector<WorkItem>,
    stealers: RwLock<Vec<Stealer<WorkItem>>>,
}

struct Split {
    row_range: Range<u64>,

    filters: Vec<Box<dyn MaskEvaluation>>,
    filter_segments: Vec<HashSet<SegmentId>>, // FIXME(ngates): BTreeSet?

    projection: Box<dyn ArrayEvaluation>,
    projection_segments: HashSet<SegmentId>,
}

enum WorkItem {
    MakeProgress { split_idx: usize },
}

struct Scheduler {
    source: Arc<dyn SegmentSource>,
    segment_futures: FuturesUnordered<BoxFuture<'static, VortexResult<IoEvent>>>,

    /// If all splits have been processed, we can stop.
    finished: bool,
    /// If any I/O request errors, we need all workers to stop with error.
    errored: bool,

    /// The range of splits that we are currently processing. All splits before should be
    /// finished, and all splits after are pending.
    active_splits: Range<usize>,
    splits: Vec<SplitState>,

    /// The working set of segments.
    working_set: Arc<DashMap<SegmentId, ByteBuffer>>,
    /// The total size of bytes in the working set of segments.
    working_set_size: u64,
    /// The target size of bytes in the working set of segments.
    target_working_set_size: u64,

    /// The filter expression for the scan.
    filter: Option<Arc<FilterExpr>>,
}

enum SplitState {
    Pending,
    Filtering {
        conjunct_idx: usize,
        waiting_for: HashSet<SegmentId>,
    },
    Projecting {
        waiting_for: HashSet<SegmentId>,
    },
    Finished,
}

struct IoEvent {
    segment_id: SegmentId,
    buffer: ByteBuffer,
}

impl Scheduler {
    /// Try to make progress scheduling I/O and CPU tasks in a non-blocking way.
    ///
    /// Returns true if progress was made.
    fn make_progress(&mut self) -> VortexResult<bool> {
        let mut made_progress = false;

        // First we attempt to handle I/O completions
        while let Poll::Ready(Some(result)) = self
            .segment_futures
            .poll_next_unpin(&mut Context::from_waker(&DUMMY_WAKER))
        {
            made_progress = true;
            match result {
                Ok(io_event) => self.handle_io_event(io_event),
                Err(e) => {
                    self.errored = true;
                    return Err(e);
                }
            }
        }

        // Spawn any split tasks that are ready to run.

        // NOTE(ngates): for worker affinity, we can create a channel per worker to send tasks
        //  down. Workers should then pull tasks from this channel and push them into their worker
        //  queue to make them eligible for stealing.

        // Now we spawn splits based on making progress with reasonable memory constraints and I/O
        // targets.
        while self.working_set_size < self.target_working_set_size {
            // self.expand_splits()
        }

        if self.active_splits.is_empty() {
            // If our range of active splits is empty, we should always attempt to make progress.
        }

        let segment_id = SegmentId::from(10);
        let segment_fut = self.source.request(segment_id);
        self.segment_futures.push(
            async move {
                let buffer = segment_fut.await?;
                Ok(IoEvent { segment_id, buffer })
            }
            .boxed(),
        );

        Ok(made_progress)
    }

    /// Block waiting for some more I/O to complete.
    fn wait_for_io(&mut self) -> VortexResult<()> {
        assert!(!self.segment_futures.is_empty(), "no in-flight I/O");
        if let Some(result) = block_on(self.segment_futures.next()) {
            match result {
                Ok(io_event) => self.handle_io_event(io_event),
                Err(e) => {
                    self.errored = true;
                    return Err(e);
                }
            }
        }
        Ok(())
    }

    fn handle_io_event(&mut self, event: IoEvent) {}
}

struct ScanWorker {
    global: Arc<GlobalState>,
    worker: Worker<WorkItem>,

    finished: bool,
    completed: VecDeque<ArrayRef>,
}

impl Iterator for ScanWorker {
    type Item = VortexResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
            // Attempt to yield any completed arrays
            if let Some(array) = self.completed.pop_front() {
                return Some(Ok(array));
            }

            // Attempt to make progress on the scheduler to spawn new I/O or handle completed I/O.
            if let Some(mut scheduler) = self.global.scheduler.try_lock() {
                if scheduler.errored {
                    return Some(Err(vortex_err!("scheduler errored, stopping worker")));
                }
                match scheduler.make_progress() {
                    Ok(made_progress) if made_progress => continue,
                    Ok(_) => {}
                    Err(e) => {
                        return Some(Err(e));
                    }
                }
            }

            // If we failed to make progress, we look to process our pending work.
            if let Some(task) = self.find_task() {
                if let Err(e) = self.handle_task(task) {
                    return Some(Err(e));
                }
                // Otherwise, continue to the next iteration of the loop.
                continue;
            }

            // Finally, we check for termination.
            if self.finished {
                return None;
            }

            // We block waiting to take a lock on the scheduler. If there's no work, but there is
            // outstanding I/O, then all workers will block here until the one worker holding the
            // lock detects I/O and releases it.
            let mut scheduler = self.global.scheduler.lock();
            // The scheduler is finished, we mark ourselves as finished and continue for one
            // more loop.
            if scheduler.finished {
                self.finished = true;
                continue;
            }
            // Otherwise, we block waiting for an I/O event while holding the scheduler lock.
            if let Err(e) = scheduler.wait_for_io() {
                return Some(Err(e));
            }
        }
    }
}

impl ScanWorker {
    fn handle_task(&mut self, work_item: WorkItem) -> VortexResult<()> {
        // TODO(ngates): should we push completed arrays into a BTree such that workers can emit
        //  the arrays in order? Possibly.
        todo!()
    }

    /// Find the next CPU task.
    fn find_task(&mut self) -> Option<WorkItem> {
        // Pop a task from the local queue, if not empty.
        self.worker.pop().or_else(|| {
            // Otherwise, we need to look for a task elsewhere.
            iter::repeat_with(|| {
                // Try stealing a batch of tasks from the global queue.
                self.global
                    .injector
                    .steal_batch_and_pop(&self.worker)
                    // Or try stealing a task from one of the other threads.
                    .or_else(|| {
                        self.global
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

static DUMMY_WAKER: LazyLock<Waker> = LazyLock::new(|| Waker::from(Arc::new(DummyWaker)));

struct DummyWaker;
impl Wake for DummyWaker {
    fn wake(self: Arc<Self>) {
        // Do nothing!
    }
}
