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
use futures::task::noop_waker;
use futures::{FutureExt, StreamExt};
use log::info;
use parking_lot::{Mutex, RwLock};
use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::{iter, mem};
use vortex_array::ArrayRef;
use vortex_array::iter::ArrayIterator;
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult, vortex_err, vortex_panic};
use vortex_layout::segments::{SegmentId, SegmentSource};
use vortex_layout::{ArrayEvaluation, MaskEvaluation};
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

pub struct Scan2 {
    global: Arc<GlobalState>,
}

struct GlobalState {
    /// The DType of the projection
    dtype: DType,
    scheduler: Mutex<Scheduler>,
    injector: Arc<Injector<Box<dyn ScanTask>>>,
    stealers: RwLock<Vec<Stealer<Box<dyn ScanTask>>>>,
}

trait TaskSpawner {
    fn spawn_task(&self, task: Box<dyn ScanTask>);
}

impl Scan2 {
    pub fn try_new(ranges: Vec<Range<u64>>, ctx: TaskContext<ArrayRef>) -> VortexResult<Self> {
        let dtype = ctx.projection.return_dtype(ctx.reader.dtype())?;

        let nsplits = ranges.len();
        let mut splits = Vec::with_capacity(nsplits);
        for row_range in ranges.into_iter() {
            let len = usize::try_from(row_range.end - row_range.start)
                .vortex_expect("Split row range is larger than usize");

            let mut filters = vec![];
            let mut filter_segments = vec![];
            if let Some(filter) = ctx.filter.as_ref() {
                for conjunct in filter.conjuncts() {
                    let eval = ctx.reader.filter_evaluation(&row_range, conjunct)?;

                    let mut segments = HashSet::new();
                    eval.required_segments(&mut segments);

                    filters.push(Some(eval));
                    filter_segments.push(segments);
                }
            }

            let projection = ctx
                .reader
                .projection_evaluation(&row_range, &ctx.projection)?;
            let mut projection_segments = HashSet::new();
            projection.required_segments(&mut projection_segments);

            splits.push(Split {
                len,
                filters,
                filter_segments,
                projection: Some(projection),
                projection_segments,
            });
        }

        let injector = Arc::new(Injector::new());

        let (cpu_send, cpu_recv) = crossbeam_channel::unbounded();

        let scheduler = Mutex::new(Scheduler {
            task_spawner: injector.clone() as _,
            filter: ctx.filter.clone(),
            source: ctx.segment_source.clone(),
            segment_futures: Default::default(),
            result_send: cpu_send,
            result_recv: cpu_recv,
            finished: false,
            errored: false,
            active_splits: 0..0,
            split_state: iter::repeat_with(|| SplitState::NotStarted)
                .take(nsplits)
                .collect(),
            splits,
            working_set: Default::default(),
            working_set_size: 0,
            target_segment_count: 1000,
            waiting_for_segments: Default::default(),
        });

        let global = Arc::new(GlobalState {
            dtype,
            scheduler,
            injector,
            stealers: Default::default(),
        });

        Ok(Self { global })
    }

    pub fn new_worker(&self) -> ScanWorker {
        let worker = Worker::new_fifo();
        self.global.stealers.write().push(worker.stealer());

        ScanWorker {
            global: self.global.clone(),
            worker,
            finished: false,
        }
    }
}

impl TaskSpawner for Injector<Box<dyn ScanTask>> {
    fn spawn_task(&self, task: Box<dyn ScanTask>) {
        self.push(task);
    }
}

struct Split {
    len: usize,

    filters: Vec<Option<Box<dyn MaskEvaluation>>>,
    filter_segments: Vec<HashSet<SegmentId>>, // FIXME(ngates): BTreeSet?

    projection: Option<Box<dyn ArrayEvaluation>>,
    projection_segments: HashSet<SegmentId>,
}

impl Debug for Split {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Split")
            .field("len", &self.len)
            .field("filter_segments", &self.filter_segments.len())
            .field("projection_segments", &self.projection_segments.len())
            .finish()
    }
}

type SplitIdx = usize;

enum Atom {
    Filter(usize),
    Project,
}

struct IoEvent {
    segment_id: SegmentId,
    buffer: ByteBuffer,
}

#[derive(Debug)]
enum ScanTaskResult {
    Filter {
        split_idx: SplitIdx,
        mask: Mask,
    },
    Project {
        split_idx: SplitIdx,
        array: ArrayRef,
    },
}

/// Scheduler for a Vortex scan.
///
/// Decides which segments to request and when, as well as spawning CPU work when available.
/// This scheduler is wrapped up for various threading models.
struct Scheduler {
    task_spawner: Arc<dyn TaskSpawner>,
    filter: Option<Arc<FilterExpr>>,
    source: Arc<dyn SegmentSource>,
    segment_futures: FuturesUnordered<BoxFuture<'static, VortexResult<IoEvent>>>,

    /// Results for scan tasks.
    result_send: Sender<ScanTaskResult>,
    result_recv: Receiver<ScanTaskResult>,

    /// If all splits have been processed, we can stop.
    finished: bool,
    /// If any I/O request errors, we need all workers to stop with error.
    errored: bool,

    /// The range of splits that we are currently processing. All splits before should be
    /// finished, and all splits after are pending.
    active_splits: Range<usize>,
    split_state: Vec<SplitState>,
    splits: Vec<Split>,

    /// The working set of segments.
    working_set: Arc<DashMap<SegmentId, ByteBuffer>>,
    /// The total size of bytes in the working set of segments.
    working_set_size: u64,

    /// The target number of segments to hold in memory.
    /// TODO(ngates): use segment size once the segment source reports it.
    target_segment_count: usize,

    /// Track which work items are waiting for segments.
    waiting_for_segments: HashMap<SegmentId, Vec<SplitIdx>>,
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

pub trait ScanTask {
    /// Execute the scan task on the current thread.
    ///
    /// If this is a projection task, the result array is returned to allow for out-of-order
    /// results for systems that are able to accept them. Otherwise, in-order results are available
    /// from the scheduler.
    fn execute(&self) -> VortexResult<Option<ArrayRef>>;
}

struct FilterTask {
    split_idx: SplitIdx,
    eval: Box<dyn MaskEvaluation>,
    mask: Mask,
    segments: Arc<DashMap<SegmentId, ByteBuffer>>,
    cpu_events: Sender<ScanTaskResult>,
}

impl ScanTask for FilterTask {
    fn execute(&self) -> VortexResult<Option<ArrayRef>> {
        let mask = self
            .eval
            .invoke(self.mask.clone(), self.segments.as_ref())?;
        self.cpu_events
            .send(ScanTaskResult::Filter {
                split_idx: self.split_idx,
                mask,
            })
            .map_err(|e| vortex_err!("failed to send project event: {}", e))?;
        Ok(None)
    }
}

struct ProjectTask {
    split_idx: SplitIdx,
    eval: Box<dyn ArrayEvaluation>,
    mask: Mask,
    segments: Arc<DashMap<SegmentId, ByteBuffer>>,
    cpu_events: Sender<ScanTaskResult>,
}

impl ScanTask for ProjectTask {
    fn execute(&self) -> VortexResult<Option<ArrayRef>> {
        let array = self
            .eval
            .invoke(self.mask.clone(), self.segments.as_ref())?;
        self.cpu_events
            .send(ScanTaskResult::Project {
                split_idx: self.split_idx,
                array: array.clone(),
            })
            .map_err(|e| vortex_err!("failed to send project event: {}", e))?;
        Ok(Some(array))
    }
}

impl Scheduler {
    /// Try to make progress scheduling I/O and CPU tasks in a non-blocking way.
    ///
    /// Returns true if progress was made.
    fn make_progress(&mut self) -> VortexResult<bool> {
        // First, we handle I/O events
        let waker = noop_waker();
        let mut ctx = Context::from_waker(&waker);
        while let Poll::Ready(Some(result)) = self.segment_futures.poll_next_unpin(&mut ctx) {
            match result {
                Ok(event) => {
                    info!("Handling I/O event for segment {:?}", event.segment_id);
                    self.handle_io_event(event)
                }
                Err(e) => {
                    self.errored = true;
                    return Err(e);
                }
            }
        }

        // Next we handle CPU events.
        while let Ok(event) = self.result_recv.try_recv() {
            info!("Handling CPU event for segment {:?}", event);
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
            if matches!(self.split_state[split_idx], SplitState::Finished) {
                info!("Completed up to split {}", split_idx);
                self.active_splits.start += 1;
            } else {
                break;
            }
        }

        // Check for termination.
        if self.active_splits.start == self.split_state.len() {
            info!("Completed all splits, terminating");
            self.finished = true;
            return Ok(made_progress);
        }

        // Now we can look to launch new splits based on the total working set size and
        // in-flight request sizes. We don't currently know the in-flight segment sizes, so we
        // will just do it based on segment count instead.
        while (self.active_splits.end < self.split_state.len()
            && self.working_set.len() + self.segment_futures.len() < self.target_segment_count)
            || self.active_splits.is_empty()
        {
            info!("Launching split {}", self.active_splits.end);
            while self.make_progress_on_split(self.active_splits.end) {}
            self.active_splits.end += 1;
            made_progress = true;
        }

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
        self.working_set.insert(event.segment_id, event.buffer);

        // Check which splits are waiting for this segment.
        if let Some(items) = self.waiting_for_segments.remove(&event.segment_id) {
            for split_idx in items {
                let split_state = &mut self.split_state[split_idx];
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
            }
        }
    }

    fn handle_cpu_event(&mut self, event: ScanTaskResult) {
        info!("Handling CPU event {:?}", event);

        match event {
            ScanTaskResult::Filter {
                split_idx, mask, ..
            } => {
                let SplitState::Filter { result, .. } = &mut self.split_state[split_idx] else {
                    vortex_panic!("unexpected split state {:?}", self.split_state[split_idx])
                };
                *result = Some(mask);
            }
            ScanTaskResult::Project { split_idx, array } => {
                let SplitState::Project { result, .. } = &mut self.split_state[split_idx] else {
                    vortex_panic!("unexpected split state {:?}", self.split_state[split_idx])
                };
                *result = Some(array);
            }
        }
    }

    fn make_progress_on_split(&mut self, split_idx: SplitIdx) -> bool {
        let split = &self.splits[split_idx];

        // Take the state temporarily, we will restore it later.
        let state = mem::replace(&mut self.split_state[split_idx], SplitState::NotStarted);

        let new_state = match &state {
            SplitState::NotStarted => {
                // We need to launch a filter task if there is one, else a project.
                let mask = Mask::new_true(split.len);
                if self.filter.is_some() {
                    Some(SplitState::PendingFilter {
                        conjunct_idx: 0,
                        mask,
                        waiting_for: self.launch_segment_requests(split_idx, Atom::Filter(0)),
                    })
                } else {
                    Some(SplitState::PendingProject {
                        mask,
                        waiting_for: self.launch_segment_requests(split_idx, Atom::Project),
                    })
                }
            }
            SplitState::PendingFilter {
                conjunct_idx,
                mask,
                waiting_for,
            } => {
                if waiting_for.is_empty() {
                    // If we have all our segments, we can launch the filter task.
                    self.task_spawner.spawn_task(Box::new(FilterTask {
                        split_idx,
                        eval: self.splits[split_idx].filters[*conjunct_idx]
                            .take()
                            .vortex_expect("filter evaluation already taken"),
                        mask: mask.clone(),
                        segments: self.working_set.clone(),
                        cpu_events: self.result_send.clone(),
                    }));
                    Some(SplitState::Filter {
                        conjunct_idx: *conjunct_idx,
                        result: None,
                    })
                } else {
                    None
                }
            }
            SplitState::Filter {
                conjunct_idx,
                result,
            } => {
                if let Some(mask) = result {
                    if mask.all_false() {
                        // If the mask is all false, we can terminate the split early.
                        Some(SplitState::Finished)
                    } else if *conjunct_idx
                        == self.filter.as_ref().map_or(0, |f| f.conjuncts().len()) - 1
                    {
                        // It was the last conjunct, so move onto projection.
                        Some(SplitState::PendingProject {
                            mask: mask.clone(),
                            waiting_for: self.launch_segment_requests(split_idx, Atom::Project),
                        })
                    } else {
                        Some(SplitState::PendingFilter {
                            conjunct_idx: conjunct_idx + 1,
                            mask: mask.clone(),
                            waiting_for: self
                                .launch_segment_requests(split_idx, Atom::Filter(conjunct_idx + 1)),
                        })
                    }
                } else {
                    None
                }
            }
            SplitState::PendingProject { mask, waiting_for } => {
                if waiting_for.is_empty() {
                    // If we have all our segments, we can launch the project task.
                    self.task_spawner.spawn_task(Box::new(ProjectTask {
                        split_idx,
                        eval: self.splits[split_idx]
                            .projection
                            .take()
                            .vortex_expect("projection evaluation already taken"),
                        mask: mask.clone(),
                        segments: self.working_set.clone(),
                        cpu_events: self.result_send.clone(),
                    }));
                    Some(SplitState::Project { result: None })
                } else {
                    None
                }
            }
            SplitState::Project { result } => {
                if result.is_some() {
                    Some(SplitState::Finished)
                } else {
                    None
                }
            }
            SplitState::Finished => {
                // We're already finished, no progress to make
                None
            }
        };

        let made_progress = new_state.is_some();
        if made_progress {
            info!(
                "Moved split {} to {:?}",
                split_idx, &self.split_state[split_idx]
            );
        }
        self.split_state[split_idx] = new_state.unwrap_or(state);
        made_progress
    }

    /// Launch requests for the given segments, returning the pending segments.
    fn launch_segment_requests(&mut self, split_idx: SplitIdx, atom: Atom) -> HashSet<SegmentId> {
        let mut pending = HashSet::new();

        let segment_ids = match atom {
            Atom::Filter(conjunct_idx) => &self.splits[split_idx].filter_segments[conjunct_idx],
            Atom::Project => &self.splits[split_idx].projection_segments,
        };

        for segment_id in segment_ids.iter().copied() {
            if self.working_set.contains_key(&segment_id) {
                continue;
            }
            let fut = self.source.request(segment_id);
            self.segment_futures.push(
                async move {
                    let buffer = fut.await?;
                    Ok(IoEvent { segment_id, buffer })
                }
                .boxed(),
            );
            pending.insert(segment_id);
            self.waiting_for_segments
                .entry(segment_id)
                .or_default()
                .push(split_idx)
        }

        pending
    }
}

pub struct ScanWorker {
    global: Arc<GlobalState>,
    worker: Worker<Box<dyn ScanTask>>,

    finished: bool,
}

impl Iterator for ScanWorker {
    type Item = VortexResult<ArrayRef>;

    fn next(&mut self) -> Option<Self::Item> {
        loop {
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
                match task.execute() {
                    // We immediately return the result since we don't care about ordering.
                    Ok(Some(array)) => return Some(Ok(array)),
                    Ok(None) => {}
                    Err(e) => {
                        return Some(Err(e));
                    }
                }
                if let Err(e) = task.execute() {
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

impl ArrayIterator for ScanWorker {
    fn dtype(&self) -> &DType {
        &self.global.dtype
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
