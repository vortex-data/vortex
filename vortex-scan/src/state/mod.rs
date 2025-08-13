// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod multithread;
mod pool;
mod segments;
mod tokio;

use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::{iter, mem};

use bit_vec::BitVec;
use futures::channel::mpsc;
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use futures::task::noop_waker;
use futures::{FutureExt, StreamExt};
use log::debug;
use vortex_array::ArrayRef;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_panic};
use vortex_layout::segments::{SegmentId, SegmentSource, Segments};
use vortex_layout::{ArrayEvaluation, MaskEvaluation, PruningEvaluation};
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::filter::FilterExpr;
use crate::state::segments::SegmentCache;
use crate::tasks::TaskContext;

pub struct Scan2 {
    ctx: TaskContext<ArrayRef>,
    splits: Vec<Split>,
    segments: SegmentCache,
}

pub(crate) trait TaskSpawner: Send {
    fn spawn_task(&self, task: Box<dyn ScanTask>);
}

impl Scan2 {
    pub fn try_new(ranges: Vec<Range<u64>>, ctx: TaskContext<ArrayRef>) -> VortexResult<Self> {
        let nsplits = ranges.len();
        let mut splits = Vec::with_capacity(nsplits);
        let mut segments = SegmentCache::default();
        for row_range in ranges.into_iter() {
            let mask = ctx.selection.row_mask(&row_range).into_mask();
            if mask.all_false() {
                // The selection has pruned this split.
                continue;
            }

            let mut pruning = None;
            let mut pruning_segments = HashSet::new();
            let mut filters = vec![];
            let mut filter_segments = vec![];
            if let Some(filter) = ctx.filter.as_ref() {
                let eval = ctx.reader.pruning_evaluation(&row_range, filter.expr())?;
                eval.required_segments(&mut pruning_segments);
                pruning = Some(eval);

                for conjunct in filter.conjuncts() {
                    let eval = ctx.reader.filter_evaluation(&row_range, conjunct)?;

                    let mut segments = HashSet::new();
                    eval.required_segments(&mut segments);

                    filters.push(Some(eval));
                    filter_segments.push(segments);
                }
            }
            let remaining_filters = BitVec::from_elem(filters.len(), true);

            let projection = ctx
                .reader
                .projection_evaluation(&row_range, &ctx.projection)?;
            let mut projection_segments = HashSet::new();
            projection.required_segments(&mut projection_segments);

            let split = Split {
                initial_mask: Some(mask),
                pruning,
                pruning_segments,
                filters,
                filter_segments,
                remaining_filters,
                projection: Some(projection),
                projection_segments,
            };

            segments.acquire(split.all_segments());
            splits.push(split);
        }

        Ok(Self {
            ctx,
            splits,
            segments,
        })
    }

    pub(crate) fn into_scheduler(self, task_spawner: Box<dyn TaskSpawner>) -> Scheduler {
        let nsplits = self.splits.len();

        // We're ok with an unbounded channel since the scheduler controls how many cpu tasks
        // are in-flight.
        let (cpu_send, cpu_recv) = mpsc::unbounded();

        Scheduler {
            task_spawner,
            filter: self.ctx.filter.clone(),
            source: self.ctx.segment_source.clone(),
            segment_futures: Default::default(),
            output_buffer: Default::default(),
            result_send: cpu_send,
            result_recv: cpu_recv,
            finished: false,
            errored: false,
            active_splits: 0..0,
            split_state: iter::repeat_with(|| SplitState::NotStarted)
                .take(nsplits)
                .collect(),
            splits: self.splits,
            segments: self.segments,
            target_segment_count: 1000,
            waiting_for_segments: Default::default(),
        }
    }
}

struct Split {
    initial_mask: Option<Mask>,

    pruning: Option<Box<dyn PruningEvaluation>>,
    pruning_segments: HashSet<SegmentId>,

    filters: Vec<Option<Box<dyn MaskEvaluation>>>,
    filter_segments: Vec<HashSet<SegmentId>>, // TODO(ngates): BTreeSet?
    remaining_filters: BitVec,

    projection: Option<Box<dyn ArrayEvaluation>>,
    projection_segments: HashSet<SegmentId>,
}

impl Debug for Split {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Split")
            .field("initial_mask", &self.initial_mask.is_some())
            .field("pruning_segments", &self.pruning_segments)
            .field("filter_segments", &self.filter_segments)
            .field("remaining_filters", &self.remaining_filters)
            .field("projection_segments", &self.projection_segments)
            .finish()
    }
}

impl Split {
    fn all_segments(&self) -> impl Iterator<Item = &SegmentId> + '_ {
        self.pruning_segments
            .iter()
            .chain(self.filter_segments.iter().flat_map(|s| s.iter()))
            .chain(self.projection_segments.iter())
    }

    /// Returns an iterator over the remaining segments in the split.
    fn remaining_segments(&self) -> impl Iterator<Item = &SegmentId> + '_ {
        // Chain together the segments for the remaining evaluations.
        self.pruning
            .is_some()
            .then(|| self.pruning_segments.iter())
            .into_iter()
            .flatten()
            .chain(
                (0..self.filters.len())
                    .filter(|idx| self.remaining_filters[*idx])
                    .flat_map(|i| self.filter_segments[i].iter()),
            )
            .chain(
                self.projection
                    .is_some()
                    .then(|| self.projection_segments.iter())
                    .into_iter()
                    .flatten(),
            )
    }

    /// Evaluations hold onto shared segment memory, so when we split is finished, we really need
    /// to drop its state.
    fn drop_evaluations(&mut self) {
        self.pruning = None;
        self.filters.clear();
        self.projection = None;
    }
}

type SplitIdx = usize;

enum Atom {
    Prune,
    Filter(usize),
    Project,
}

struct IoEvent {
    segment_id: SegmentId,
    buffer: ByteBuffer,
}

enum ScanTaskResult {
    Prune {
        split_idx: SplitIdx,
        mask: VortexResult<Mask>,
    },
    Filter {
        split_idx: SplitIdx,
        mask: VortexResult<Mask>,
    },
    Project {
        split_idx: SplitIdx,
        array: VortexResult<ArrayRef>,
    },
}

impl Debug for ScanTaskResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ScanTaskResult")
            .field(
                "type",
                &match self {
                    ScanTaskResult::Prune { .. } => "Prune",
                    ScanTaskResult::Filter { .. } => "Filter",
                    ScanTaskResult::Project { .. } => "Project",
                },
            )
            .field(
                "split_idx",
                &match self {
                    ScanTaskResult::Prune { split_idx, .. } => *split_idx,
                    ScanTaskResult::Filter { split_idx, .. } => *split_idx,
                    ScanTaskResult::Project { split_idx, .. } => *split_idx,
                },
            )
            .finish()
    }
}

/// Scheduler for a Vortex scan.
///
/// Decides which segments to request and when, as well as spawning CPU work when available.
/// This scheduler is wrapped up for various threading models.
pub(crate) struct Scheduler {
    task_spawner: Box<dyn TaskSpawner>,
    filter: Option<Arc<FilterExpr>>,

    source: Arc<dyn SegmentSource>,
    segment_futures: FuturesUnordered<BoxFuture<'static, VortexResult<IoEvent>>>,

    /// A buffer for the arrays emitted by the scan. This buffer is used to return arrays with
    /// a total ordering over the splits of the scan.
    output_buffer: VecDeque<VortexResult<ArrayRef>>,

    /// Results for scan tasks.
    result_send: mpsc::UnboundedSender<ScanTaskResult>,
    result_recv: mpsc::UnboundedReceiver<ScanTaskResult>,

    /// If all splits have been processed, we can stop.
    finished: bool,
    /// If any I/O request errors, we want all workers to stop with error.
    errored: bool,

    /// The range of splits that we are currently processing. All splits before should be
    /// finished, and all splits after are pending.
    active_splits: Range<usize>,
    split_state: Vec<SplitState>,
    splits: Vec<Split>,

    // The segment cache used to power the scan.
    segments: SegmentCache,

    /// The target number of segments to hold in memory.
    /// TODO(ngates): use segment size once the segment source reports it.
    target_segment_count: usize,

    /// Track which work items are waiting for segments.
    waiting_for_segments: HashMap<SegmentId, Vec<SplitIdx>>,
}

#[derive(Debug)]
enum SplitState {
    NotStarted,
    StartPrune {
        mask: Mask,
    },
    PendingPrune {
        mask: Mask,
        waiting_for: HashSet<SegmentId>,
    },
    Prune {
        result: Option<Mask>,
    },
    StartFilter {
        mask: Mask,
    },
    PendingFilter {
        conjunct_idx: usize,
        mask: Mask,
        waiting_for: HashSet<SegmentId>,
    },
    Filter {
        conjunct_idx: usize,
        input_mask: Mask,
        result: Option<Mask>,
    },
    StartProject {
        mask: Mask,
    },
    PendingProject {
        mask: Mask,
        waiting_for: HashSet<SegmentId>,
    },
    Project {
        result: Option<ArrayRef>,
    },
    Errored(Arc<VortexError>),
    Finished,
}

pub trait ScanTask: Send {
    /// Execute the scan task on the current thread.
    ///
    /// If this is a projection task, the result array is returned to allow for out-of-order
    /// results for systems that are able to accept them. Otherwise, in-order results are available
    /// from the scheduler.
    fn execute(&self) -> Option<ArrayRef>;
}

struct PruneTask {
    split_idx: SplitIdx,
    eval: Box<dyn PruningEvaluation>,
    mask: Mask,
    segments: Arc<dyn Segments>,
    cpu_events: mpsc::UnboundedSender<ScanTaskResult>,
}

impl ScanTask for PruneTask {
    fn execute(&self) -> Option<ArrayRef> {
        let mask = self.eval.invoke(self.mask.clone(), self.segments.as_ref());
        // Ignore the error, since it means the scan has terminated early.
        let _ = self.cpu_events.unbounded_send(ScanTaskResult::Prune {
            split_idx: self.split_idx,
            mask,
        });
        None
    }
}

struct FilterTask {
    // TODO(ngates): we may wish to plumb through an is_canceled: Arc<AtomicBool>.
    split_idx: SplitIdx,
    eval: Box<dyn MaskEvaluation>,
    mask: Mask,
    segments: Arc<dyn Segments>,
    cpu_events: mpsc::UnboundedSender<ScanTaskResult>,
}

impl ScanTask for FilterTask {
    fn execute(&self) -> Option<ArrayRef> {
        let mask = self.eval.invoke(self.mask.clone(), self.segments.as_ref());
        // Ignore the error, since it means the scan has terminated early.
        let _ = self.cpu_events.unbounded_send(ScanTaskResult::Filter {
            split_idx: self.split_idx,
            mask,
        });
        None
    }
}

struct ProjectTask {
    split_idx: SplitIdx,
    eval: Box<dyn ArrayEvaluation>,
    mask: Mask,
    segments: Arc<dyn Segments>,
    cpu_events: mpsc::UnboundedSender<ScanTaskResult>,
}

impl ScanTask for ProjectTask {
    fn execute(&self) -> Option<ArrayRef> {
        let array = self.eval.invoke(self.mask.clone(), self.segments.as_ref());

        // We take a (zero-)copy of the array for scan drivers that are able to immediately return
        // out-of-order results.
        let result = array.as_ref().ok().cloned();

        // Ignore the error, since it means the scan has terminated early.
        let _ = self.cpu_events.unbounded_send(ScanTaskResult::Project {
            split_idx: self.split_idx,
            array,
        });
        result
    }
}

impl Scheduler {
    /// Try to make progress scheduling I/O and CPU tasks in a non-blocking way.
    ///
    /// Returns true if progress was made.
    fn make_progress(&mut self) -> Poll<VortexResult<()>> {
        let waker = noop_waker();
        let mut ctx = Context::from_waker(&waker);
        self.make_progress_with_cx(&mut ctx)
    }

    fn make_progress_with_cx(&mut self, cx: &mut Context) -> Poll<VortexResult<()>> {
        let mut made_progress = false;

        // First, we handle I/O events
        while let Poll::Ready(Some(result)) = self.segment_futures.poll_next_unpin(cx) {
            made_progress = true;
            match result {
                Ok(event) => self.handle_io_event(event),
                Err(e) => {
                    self.errored = true;
                    return Poll::Ready(Err(e));
                }
            }
        }

        // Next we handle CPU events.
        while let Poll::Ready(Some(event)) = self.result_recv.poll_next_unpin(cx) {
            made_progress = true;
            self.handle_cpu_event(event);
        }

        // We bring forward the start of the active splits if any splits are finished.
        for split_idx in self.active_splits.clone() {
            if matches!(self.split_state[split_idx], SplitState::Finished) {
                debug!("Completed splits {}/{}", split_idx, self.split_state.len());
                self.active_splits.start += 1;
            } else {
                break;
            }
        }

        // Now we can look to launch new splits based on the total working set size and
        // in-flight request sizes. We don't currently know the in-flight segment sizes, so we
        // will just do it based on segment count instead.
        while self.active_splits.end < self.split_state.len()
            && ((self.segments.len() + self.segment_futures.len() < self.target_segment_count)
                || self.active_splits.is_empty())
        {
            debug!("Initializing split {}", self.active_splits.end);
            made_progress = true;
            self.make_progress_on_split(self.active_splits.end);
            self.active_splits.end += 1;
        }

        // Check for termination.
        if self.active_splits.start == self.split_state.len() {
            self.finished = true;
            if !self.segments.is_empty() {
                log::warn!(
                    "Unreleased segments: {:?}\n{:?}",
                    self.segments.ref_counts(),
                    self.splits
                );
                vortex_panic!("Unreleased segments. Bug in reference counting.")
            }
        }

        if made_progress {
            Poll::Ready(Ok(()))
        } else {
            Poll::Pending
        }
    }

    /// Handles an I/O event by updating the splits that are waiting for this particular segment.
    /// Finishes by driving the state machines of any impacted splits.
    fn handle_io_event(&mut self, event: IoEvent) {
        // Insert the segment buffer into the working set.
        self.segments.insert(event.segment_id, event.buffer);

        // Check which splits are waiting for this segment.
        if let Some(items) = self.waiting_for_segments.remove(&event.segment_id) {
            for split_idx in items {
                let split_state = &mut self.split_state[split_idx];
                match split_state {
                    SplitState::PendingPrune { waiting_for, .. } => {
                        waiting_for.remove(&event.segment_id);
                    }
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
                self.make_progress_on_split(split_idx);
            }
        }
    }

    /// Handles a CPU event by updating the split state based on the result of a filter or
    /// project task.
    ///
    /// Finishes by driving the state machines of any impacted splits.
    fn handle_cpu_event(&mut self, event: ScanTaskResult) {
        match event {
            ScanTaskResult::Prune { split_idx, mask } => {
                match mask {
                    Ok(mask) => {
                        let SplitState::Prune { result, .. } = &mut self.split_state[split_idx]
                        else {
                            vortex_panic!(
                                "unexpected split state {:?}",
                                self.split_state[split_idx]
                            )
                        };
                        *result = Some(mask);
                    }
                    Err(e) => {
                        self.split_state[split_idx] = SplitState::Errored(Arc::new(e));
                    }
                }
                self.make_progress_on_split(split_idx);
            }
            ScanTaskResult::Filter {
                split_idx, mask, ..
            } => {
                match mask {
                    Ok(mask) => {
                        let SplitState::Filter { result, .. } = &mut self.split_state[split_idx]
                        else {
                            vortex_panic!(
                                "unexpected split state {:?}",
                                self.split_state[split_idx]
                            )
                        };
                        *result = Some(mask);
                    }
                    Err(e) => {
                        self.split_state[split_idx] = SplitState::Errored(Arc::new(e));
                    }
                }
                self.make_progress_on_split(split_idx);
            }
            ScanTaskResult::Project { split_idx, array } => {
                match array {
                    Ok(array) => {
                        let SplitState::Project { result, .. } = &mut self.split_state[split_idx]
                        else {
                            vortex_panic!(
                                "unexpected split state {:?}",
                                self.split_state[split_idx]
                            )
                        };
                        *result = Some(array);
                    }
                    Err(e) => {
                        self.split_state[split_idx] = SplitState::Errored(Arc::new(e));
                    }
                }
                self.make_progress_on_split(split_idx);
            }
        }
    }

    fn make_progress_on_split(&mut self, split_idx: SplitIdx) {
        // We sit in a loop so that we drive the split as far as possible. For example, if we
        // enter the PendingFilter state, but all segments are already resolved, we can perform
        // another state transition immediately. It simplifies the logic to keep these transitions
        // separate and run in a loop.
        loop {
            // Take the state temporarily, we will restore it later.
            let state = mem::replace(&mut self.split_state[split_idx], SplitState::NotStarted);

            let new_state = match &state {
                SplitState::NotStarted => {
                    // We need to launch a filter task if there is one, else a project.
                    let mask = self.splits[split_idx]
                        .initial_mask
                        .take()
                        .vortex_expect("Initial mask already taken");
                    if self.filter.is_some() {
                        Some(SplitState::StartPrune { mask })
                    } else {
                        Some(SplitState::StartProject { mask })
                    }
                }
                SplitState::StartPrune { mask } => Some(SplitState::PendingPrune {
                    mask: mask.clone(),
                    waiting_for: self.launch_segment_requests(split_idx, Atom::Prune),
                }),
                SplitState::PendingPrune { mask, waiting_for } => {
                    waiting_for.is_empty().then(|| {
                        debug!("Spawning pruning for split: {}", split_idx);
                        self.task_spawner.spawn_task(Box::new(PruneTask {
                            split_idx,
                            eval: self.splits[split_idx]
                                .pruning
                                .take()
                                .vortex_expect("pruning evaluation already taken"),
                            mask: mask.clone(),
                            segments: self.segments.segments(),
                            cpu_events: self.result_send.clone(),
                        }));
                        SplitState::Prune { result: None }
                    })
                }
                SplitState::Prune { result } => {
                    if let Some(mask) = result {
                        // Release the pruning segments.
                        self.segments
                            .release(self.splits[split_idx].pruning_segments.iter());
                        Some(SplitState::StartFilter { mask: mask.clone() })
                    } else {
                        None
                    }
                }
                SplitState::StartFilter { mask } => {
                    if mask.all_false() {
                        // If we have an all-false mask, we can finish the split without projecting.
                        self.segments
                            .release(self.splits[split_idx].remaining_segments());
                        self.splits[split_idx].drop_evaluations();
                        Some(SplitState::Finished)
                    } else {
                        match self.filter.as_ref().and_then(|f| {
                            f.next_conjunct(&self.splits[split_idx].remaining_filters)
                        }) {
                            None => {
                                // No more conjuncts to invoke.
                                Some(SplitState::StartProject { mask: mask.clone() })
                            }
                            Some(conjunct_idx) => {
                                // Mark the conjunct as used.
                                self.splits[split_idx]
                                    .remaining_filters
                                    .set(conjunct_idx, false);

                                Some(SplitState::PendingFilter {
                                    conjunct_idx,
                                    mask: mask.clone(),
                                    waiting_for: self.launch_segment_requests(
                                        split_idx,
                                        Atom::Filter(conjunct_idx),
                                    ),
                                })
                            }
                        }
                    }
                }
                SplitState::PendingFilter {
                    conjunct_idx,
                    mask,
                    waiting_for,
                } => waiting_for.is_empty().then(|| {
                    debug!(
                        "Spawning filter for split: {}, conjunct: {}",
                        split_idx, conjunct_idx
                    );
                    self.task_spawner.spawn_task(Box::new(FilterTask {
                        split_idx,
                        eval: self.splits[split_idx].filters[*conjunct_idx]
                            .take()
                            .vortex_expect("filter evaluation already taken"),
                        mask: mask.clone(),
                        segments: self.segments.segments(),
                        cpu_events: self.result_send.clone(),
                    }));
                    SplitState::Filter {
                        conjunct_idx: *conjunct_idx,
                        input_mask: mask.clone(),
                        result: None,
                    }
                }),
                SplitState::Filter {
                    conjunct_idx,
                    input_mask,
                    result,
                } => {
                    if let Some(mask) = result {
                        self.segments
                            .release(self.splits[split_idx].filter_segments[*conjunct_idx].iter());

                        // Report the selectivity of this conjunct.
                        // TODO(ngates): what selectivity should we report?
                        let selectivity = mask.true_count() as f64 / input_mask.len() as f64;
                        // let selectivity = mask.true_count() as f64 / input_mask.true_count() as f64;
                        self.filter
                            .as_ref()
                            .vortex_expect("missing filter")
                            .report_selectivity(*conjunct_idx, selectivity);

                        Some(SplitState::StartFilter { mask: mask.clone() })
                    } else {
                        None
                    }
                }
                SplitState::StartProject { mask } => Some(SplitState::PendingProject {
                    mask: mask.clone(),
                    waiting_for: self.launch_segment_requests(split_idx, Atom::Project),
                }),
                SplitState::PendingProject { mask, waiting_for } => {
                    waiting_for.is_empty().then(|| {
                        debug!("Spawning projection for split: {}", split_idx);
                        self.task_spawner.spawn_task(Box::new(ProjectTask {
                            split_idx,
                            eval: self.splits[split_idx]
                                .projection
                                .take()
                                .vortex_expect("projection evaluation already taken"),
                            mask: mask.clone(),
                            segments: self.segments.segments(),
                            cpu_events: self.result_send.clone(),
                        }));
                        SplitState::Project { result: None }
                    })
                }
                SplitState::Project { result } => {
                    if let Some(array) = result {
                        self.segments
                            .release(self.splits[split_idx].projection_segments.iter());
                        self.output_buffer.push_back(Ok(array.clone()));
                        Some(SplitState::Finished)
                    } else {
                        None
                    }
                }
                SplitState::Errored(e) => {
                    self.output_buffer
                        .push_back(Err(VortexError::from(e.clone())));
                    self.segments
                        .release(self.splits[split_idx].remaining_segments());
                    self.splits[split_idx].drop_evaluations();
                    Some(SplitState::Finished)
                }
                SplitState::Finished => {
                    // We're already finished, no progress to make
                    None
                }
            };

            let made_progress = new_state.is_some();
            if let Some(new_state) = &new_state {
                debug!(
                    "Updated split {} from {:?} to {:?}",
                    split_idx, &state, &new_state
                );
            }
            self.split_state[split_idx] = new_state.unwrap_or(state);

            if !made_progress {
                // If we did not make progress, we can stop processing this split.
                break;
            }
        }
    }

    /// Launch requests for the given segments, returning the pending segments.
    fn launch_segment_requests(&mut self, split_idx: SplitIdx, atom: Atom) -> HashSet<SegmentId> {
        let mut pending = HashSet::new();

        let segment_ids = match atom {
            Atom::Prune => &self.splits[split_idx].pruning_segments,
            Atom::Filter(conjunct_idx) => &self.splits[split_idx].filter_segments[conjunct_idx],
            Atom::Project => &self.splits[split_idx].projection_segments,
        };

        for segment_id in segment_ids.iter().copied() {
            if self.segments.contains(&segment_id) {
                continue;
            }
            debug!("Requesting segment {:?}", segment_id);
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
