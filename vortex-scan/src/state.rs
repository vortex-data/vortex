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
use log::info;
use parking_lot::{Mutex, RwLock};
use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::sync::{Arc, LazyLock};
use std::task::{Context, Poll, Wake, Waker};
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
    scheduler: Mutex<Scheduler>,
    shared: Arc<SharedState>,
    stealers: RwLock<Vec<Stealer<Box<dyn ScanTask>>>>,
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

        let shared = Arc::new(SharedState {
            dtype,
            splits: vec![],
            injector: Injector::new(),
            working_set: Default::default(),
            filter_expr: ctx.filter.clone(),
        });

        let (cpu_send, cpu_recv) = crossbeam_channel::unbounded();

        let scheduler = Mutex::new(Scheduler {
            shared: shared.clone(),
            source: ctx.segment_source.clone(),
            segment_futures: Default::default(),
            cpu_send,
            cpu_recv,
            finished: false,
            errored: false,
            active_splits: 0..0,
            split_state: iter::repeat_with(|| SplitState::NotStarted)
                .take(nsplits)
                .collect(),
            splits,
            working_set_size: 0,
            target_segment_count: 1000,
            waiting_for_segments: Default::default(),
        });

        let global = Arc::new(GlobalState {
            scheduler,
            shared,
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

/// State that is shared across workers and the scheduler
struct SharedState {
    /// The DType of the projection
    dtype: DType,
    /// Information about each split in the scan.
    splits: Vec<Split>,
    /// Injector for launching CPU tasks.
    injector: Injector<Box<dyn ScanTask>>,
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
    cpu_send: Sender<CpuEvent>,
    cpu_recv: Receiver<CpuEvent>,

    /// If all splits have been processed, we can stop.
    finished: bool,
    /// If any I/O request errors, we need all workers to stop with error.
    errored: bool,

    /// The range of splits that we are currently processing. All splits before should be
    /// finished, and all splits after are pending.
    active_splits: Range<usize>,
    split_state: Vec<SplitState>,
    splits: Vec<Split>,

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
    conjunct_idx: usize,
    eval: Box<dyn MaskEvaluation>,
    mask: Mask,
    segments: Arc<DashMap<SegmentId, ByteBuffer>>,
    cpu_events: Sender<CpuEvent>,
}

impl ScanTask for FilterTask {
    fn execute(&self) -> VortexResult<Option<ArrayRef>> {
        let mask = self
            .eval
            .invoke(self.mask.clone(), self.segments.as_ref())?;
        self.cpu_events
            .send(CpuEvent::Filter {
                split_idx: self.split_idx,
                conjunct_idx: self.conjunct_idx,
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
    cpu_events: Sender<CpuEvent>,
}

impl ScanTask for ProjectTask {
    fn execute(&self) -> VortexResult<Option<ArrayRef>> {
        let array = self
            .eval
            .invoke(self.mask.clone(), self.segments.as_ref())?;
        self.cpu_events
            .send(CpuEvent::Project {
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
        while let Poll::Ready(Some(result)) = self
            .segment_futures
            .poll_next_unpin(&mut Context::from_waker(&DUMMY_WAKER))
        {
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
        while let Ok(event) = self.cpu_recv.try_recv() {
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
            && self.shared.working_set.len() + self.segment_futures.len()
                < self.target_segment_count)
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
        self.shared
            .working_set
            .insert(event.segment_id, event.buffer);

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

    fn handle_cpu_event(&mut self, event: CpuEvent) {
        info!("Handling CPU event {:?}", event);

        match event {
            CpuEvent::Filter {
                split_idx, mask, ..
            } => {
                let SplitState::Filter { result, .. } = &mut self.split_state[split_idx] else {
                    vortex_panic!("unexpected split state {:?}", self.split_state[split_idx])
                };
                *result = Some(mask);
            }
            CpuEvent::Project { split_idx, array } => {
                let SplitState::Project { result, .. } = &mut self.split_state[split_idx] else {
                    vortex_panic!("unexpected split state {:?}", self.split_state[split_idx])
                };
                *result = Some(array);
            }
        }
    }

    fn make_progress_on_split(&mut self, split_idx: SplitIdx) -> bool {
        let split = &self.shared.splits[split_idx];

        // Take the state temporarily, we will restore it later.
        let state = mem::replace(&mut self.split_state[split_idx], SplitState::NotStarted);

        let made_progress = match state {
            SplitState::NotStarted => {
                // We need to launch a filter task if there is one, else a project.
                let mask = Mask::new_true(split.len);
                if self.shared.filter_expr.is_some() {
                    self.split_state[split_idx] = SplitState::PendingFilter {
                        conjunct_idx: 0,
                        mask,
                        waiting_for: self.launch_segment_requests(split_idx, Atom::Filter(0)),
                    };
                } else {
                    self.split_state[split_idx] = SplitState::PendingProject {
                        mask,
                        waiting_for: self.launch_segment_requests(split_idx, Atom::Project),
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
                    self.shared.injector.push(Box::new(FilterTask {
                        split_idx,
                        eval: self.splits[split_idx].filters[conjunct_idx]
                            .take()
                            .vortex_expect("filter evaluation already taken"),
                        conjunct_idx,
                        mask: mask.clone(),
                        segments: self.shared.working_set.clone(),
                        cpu_events: self.cpu_send.clone(),
                    }));
                    self.split_state[split_idx] = SplitState::Filter {
                        conjunct_idx,
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
                        self.split_state[split_idx] = SplitState::Finished;
                        true;
                    } else if conjunct_idx == self.shared.num_conjuncts() - 1 {
                        // It was the last conjunct, so move onto projection.
                        self.split_state[split_idx] = SplitState::PendingProject {
                            mask: mask.clone(),
                            waiting_for: self.launch_segment_requests(split_idx, Atom::Project),
                        }
                    } else {
                        self.split_state[split_idx] = SplitState::PendingFilter {
                            conjunct_idx: conjunct_idx + 1,
                            mask: mask.clone(),
                            waiting_for: self
                                .launch_segment_requests(split_idx, Atom::Filter(conjunct_idx + 1)),
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
                    self.shared.injector.push(Box::new(ProjectTask {
                        split_idx,
                        eval: self.splits[split_idx]
                            .projection
                            .take()
                            .vortex_expect("projection evaluation already taken"),
                        mask: mask.clone(),
                        segments: self.shared.working_set.clone(),
                        cpu_events: self.cpu_send.clone(),
                    }));
                    self.split_state[split_idx] = SplitState::Project { result: None };
                    true
                } else {
                    false
                }
            }
            SplitState::Project { result } => {
                if result.is_some() {
                    self.split_state[split_idx] = SplitState::Finished;
                    true
                } else {
                    false
                }
            }
            SplitState::Finished => {
                // We're already finished, no progress to make
                false
            }
        };

        info!(
            "Moved split {} to {:?}",
            split_idx, &self.split_state[split_idx]
        );
        made_progress
    }

    /// Launch requests for the given segments, returning the pending segments.
    fn launch_segment_requests(&mut self, split_idx: SplitIdx, atom: Atom) -> HashSet<SegmentId> {
        let mut pending = HashSet::new();

        let segment_ids = match atom {
            Atom::Filter(conjunct_idx) => {
                &self.shared.splits[split_idx].filter_segments[conjunct_idx]
            }
            Atom::Project => &self.shared.splits[split_idx].projection_segments,
        };

        for segment_id in segment_ids.iter().copied() {
            if self.shared.working_set.contains_key(&segment_id) {
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
        &self.global.shared.dtype
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
