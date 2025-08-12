// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::filter::FilterExpr;
use crate::tasks::TaskContext;
use crossbeam_channel::{Receiver, Sender};
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
use vortex_error::{VortexResult, vortex_err, vortex_panic};
use vortex_layout::segments::{SegmentId, SegmentSource};
use vortex_layout::{ArrayEvaluation, MaskEvaluation};
use vortex_mask::Mask;
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
    shared: Arc<SharedState>,
    cpu_events: Sender<CpuEvent>,
    stealers: RwLock<Vec<Stealer<WorkItem>>>,
}

/// State that is shared across workers and the scheduler
struct SharedState {
    /// Information about each split in the scan.
    splits: Vec<Split>,
    /// Injector for launching CPU tasks.
    injector: Injector<WorkItem>,
    /// The working set of segments.
    working_set: Arc<DashMap<SegmentId, ByteBuffer>>,
    /// The filter expression for the scan.
    filter_expr: Option<Arc<FilterExpr>>,
}

impl SharedState {
    fn num_conjuncts(&self) -> usize {
        self.filter_expr.as_ref().map_or(0, |f| f.conjuncts().len())
    }
}

struct Split {
    row_range: Range<u64>,
    len: usize,

    filters: Vec<Box<dyn MaskEvaluation>>,
    filter_segments: Vec<HashSet<SegmentId>>, // FIXME(ngates): BTreeSet?

    projection: Box<dyn ArrayEvaluation>,
    projection_segments: HashSet<SegmentId>,
}

type SplitIdx = usize;

struct IoEvent {
    segment_id: SegmentId,
    buffer: ByteBuffer,
}
enum CpuEvent {
    Filter {
        split_idx: SplitIdx,
        conjunct_idx: usize,
        mask: Mask,
    },
    Project {
        split_idx: SplitIdx,
        array: ArrayRef,
    },
}

struct Scheduler {
    shared: Arc<SharedState>,

    source: Arc<dyn SegmentSource>,
    segment_futures: FuturesUnordered<BoxFuture<'static, VortexResult<IoEvent>>>,

    /// Results for CPU tasks.
    cpu_events: Receiver<CpuEvent>,

    /// If all splits have been processed, we can stop.
    finished: bool,
    /// If any I/O request errors, we need all workers to stop with error.
    errored: bool,

    /// The range of splits that we are currently processing. All splits before should be
    /// finished, and all splits after are pending.
    active_splits: Range<usize>,
    splits: Vec<SplitState>,

    /// The total size of bytes in the working set of segments.
    working_set_size: u64,
    /// The target size of bytes in the working set of segments.
    target_working_set_size: u64,

    /// Track which work items are waiting for segments.
    waiting_for_segments: Arc<DashMap<SegmentId, Vec<SplitIdx>>>,
}

#[derive(Debug)]
enum SplitState {
    NotStarted,
    PendingFilter {
        conjunct_idx: usize,
        mask: Mask,
        waiting_for: HashSet<SegmentId>,
    },
    Filter {
        conjunct_idx: usize,
        result: Option<Mask>,
    },
    PendingProject {
        mask: Mask,
        waiting_for: HashSet<SegmentId>,
    },
    Project {
        result: Option<ArrayRef>,
    },
    Finished,
}

enum WorkItem {
    Filter {
        split_idx: SplitIdx,
        conjunct_idx: usize,
        mask: Mask,
    },
    Project {
        split_idx: SplitIdx,
        mask: Mask,
    },
}

impl Scheduler {
    /// Try to make progress scheduling I/O and CPU tasks in a non-blocking way.
    ///
    /// Returns true if progress was made.
    fn make_progress(&mut self) -> VortexResult<bool> {
        // First we handle I/O events
        while let Poll::Ready(Some(result)) = self
            .segment_futures
            .poll_next_unpin(&mut Context::from_waker(&DUMMY_WAKER))
        {
            match result {
                Ok(io_event) => self.handle_io_event(io_event),
                Err(e) => {
                    self.errored = true;
                    return Err(e);
                }
            }
        }

        // Next we handle CPU events.
        while let Ok(event) = self.cpu_events.try_recv() {
            self.handle_cpu_event(event)
        }

        // Now we drive forwards the split state machines.
        let mut made_progress = false;
        for split_idx in self.active_splits.clone() {
            if self.make_progress_on_split(split_idx) {
                made_progress = true;
            }
        }

        // We bring forward the start of the active splits if any splits are finished.
        for split_idx in self.active_splits.clone() {
            if matches!(self.splits[split_idx], SplitState::Finished) {
                self.active_splits.start += 1;
            } else {
                break;
            }
        }

        // Now we can look to launch new splits based on the total working set size and
        // in-flight request sizes.

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

    fn handle_io_event(&mut self, event: IoEvent) {
        // Insert the segment buffer into the working set.
        self.working_set_size += event.buffer.len() as u64;
        self.shared
            .working_set
            .insert(event.segment_id, event.buffer);

        // Check which splits are waiting for this segment.
        if let Some((_id, items)) = self.waiting_for_segments.remove(&event.segment_id) {
            for split_idx in items {
                let split_state = &mut self.splits[split_idx];
                match split_state {
                    SplitState::PendingFilter { waiting_for, .. } => {
                        waiting_for.remove(&event.segment_id);
                    }
                    SplitState::PendingProject { waiting_for, .. } => {
                        waiting_for.remove(&event.segment_id);
                    }
                    _ => {
                        vortex_panic!("Unexpected split state {:?}", split_state)
                    }
                }
                while self.make_progress_on_split(split_idx) {}
            }
        }
    }

    fn handle_cpu_event(&mut self, event: CpuEvent) {
        match event {
            CpuEvent::Filter {
                split_idx, mask, ..
            } => {
                let SplitState::Filter { result, .. } = &mut self.splits[split_idx] else {
                    vortex_panic!("unexpected split state {:?}", self.splits[split_idx])
                };
                *result = Some(mask);
                while self.make_progress_on_split(split_idx) {}
            }
            CpuEvent::Project { split_idx, array } => {
                let SplitState::Project { result, .. } = &mut self.splits[split_idx] else {
                    vortex_panic!("unexpected split state {:?}", self.splits[split_idx])
                };
                *result = Some(array);
                while self.make_progress_on_split(split_idx) {}
            }
        }
    }

    fn make_progress_on_split(&mut self, split_idx: SplitIdx) -> bool {
        let split = &self.shared.splits[split_idx];
        let state = &mut self.splits[split_idx];
        match state {
            SplitState::NotStarted => {
                // We need to launch a filter task if there is one, else a project.
                let mask = Mask::new_true(split.len);
                if self.shared.filter_expr.is_some() {
                    *state = SplitState::PendingFilter {
                        conjunct_idx: 0,
                        mask,
                        waiting_for: split.filter_segments[0].clone(),
                    }
                } else {
                    *state = SplitState::PendingProject {
                        mask,
                        waiting_for: split.projection_segments.clone(),
                    };
                }
                true
            }
            SplitState::PendingFilter {
                conjunct_idx,
                mask,
                waiting_for,
            } => {
                if waiting_for.is_empty() {
                    // If we have all our segments, we can launch the filter task.
                    self.shared.injector.push(WorkItem::Filter {
                        split_idx,
                        conjunct_idx: *conjunct_idx,
                        mask: mask.clone(),
                    });
                    *state = SplitState::Filter {
                        conjunct_idx: *conjunct_idx,
                        result: None,
                    };
                    true
                } else {
                    false
                }
            }
            SplitState::Filter {
                conjunct_idx,
                result,
            } => {
                if let Some(mask) = result {
                    if mask.all_false() {
                        // If the mask is all false, we can terminate the split early.
                        *state = SplitState::Finished;
                        return true;
                    }

                    if *conjunct_idx == self.shared.num_conjuncts() - 1 {
                        // It was the last conjunct, so move onto projection.
                        *state = SplitState::PendingProject {
                            mask: mask.clone(),
                            waiting_for: split.projection_segments.clone(),
                        }
                    } else {
                        *state = SplitState::PendingFilter {
                            conjunct_idx: *conjunct_idx + 1,
                            mask: mask.clone(),
                            waiting_for: split.filter_segments[*conjunct_idx + 1].clone(),
                        }
                    }
                    true
                } else {
                    false
                }
            }
            SplitState::PendingProject { mask, waiting_for } => {
                if waiting_for.is_empty() {
                    // If we have all our segments, we can launch the project task.
                    self.shared.injector.push(WorkItem::Project {
                        split_idx,
                        mask: mask.clone(),
                    });
                    *state = SplitState::Project { result: None };
                    true
                } else {
                    false
                }
            }
            SplitState::Project { result } => {
                if result.is_some() {
                    *state = SplitState::Finished;
                    true
                } else {
                    false
                }
            }
            SplitState::Finished => {
                // We're already finished, no progress to make
                false
            }
        }
    }
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
                if let Err(e) = self.handle_work_item(task) {
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
    fn handle_work_item(&mut self, work_item: WorkItem) -> VortexResult<()> {
        match work_item {
            WorkItem::Filter {
                split_idx,
                mask,
                conjunct_idx,
            } => {
                let cpu_event = self.do_filter(split_idx, conjunct_idx, mask)?;
                self.global
                    .cpu_events
                    .send(cpu_event)
                    .map_err(|e| vortex_err!("failed to send CPU event: {}", e))?;
                Ok(())
            }
            WorkItem::Project { split_idx, mask } => self.do_project(split_idx, mask),
        }
    }

    fn do_filter(
        &mut self,
        split_idx: usize,
        conjunct_idx: usize,
        mask: Mask,
    ) -> VortexResult<CpuEvent> {
        let split = &self.global.shared.splits[split_idx];
        let mask =
            split.filters[conjunct_idx].invoke(mask, self.global.shared.working_set.as_ref())?;
        Ok(CpuEvent::Filter {
            split_idx,
            conjunct_idx,
            mask,
        })
    }

    fn do_project(&mut self, split_idx: usize, mask: Mask) -> VortexResult<()> {
        let split = &self.global.shared.splits[split_idx];
        let array = split
            .projection
            .invoke(mask, self.global.shared.working_set.as_ref())?;
        self.completed.push_back(array);
        Ok(())
    }

    /// Find the next CPU task.
    fn find_task(&mut self) -> Option<WorkItem> {
        // Pop a task from the local queue, if not empty.
        self.worker.pop().or_else(|| {
            // Otherwise, we need to look for a task elsewhere.
            iter::repeat_with(|| {
                // Try stealing a batch of tasks from the global queue.
                self.global
                    .shared
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
