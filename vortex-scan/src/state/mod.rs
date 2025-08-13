// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod multithread;
mod pool;
mod tokio;

use crate::filter::FilterExpr;
use crate::tasks::TaskContext;
use dashmap::DashMap;
use futures::channel::mpsc;
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use futures::task::noop_waker;
use futures::{FutureExt, Stream, StreamExt};
use log::{debug, info};
use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::{iter, mem};
use vortex_array::ArrayRef;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexError, VortexExpect, VortexResult, vortex_panic};
use vortex_layout::segments::{SegmentId, SegmentSource};
use vortex_layout::{ArrayEvaluation, MaskEvaluation};
use vortex_mask::Mask;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

pub struct Scan2 {
    ctx: TaskContext<ArrayRef>,
    splits: Vec<Split>,
}

pub(crate) trait TaskSpawner: Send {
    fn spawn_task(&self, task: Box<dyn ScanTask>);
}

impl Scan2 {
    pub fn try_new(ranges: Vec<Range<u64>>, ctx: TaskContext<ArrayRef>) -> VortexResult<Self> {
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

        Ok(Self { ctx, splits })
    }

    pub(crate) fn into_scheduler(self, task_spawner: Box<dyn TaskSpawner>) -> Scheduler {
        let nsplits = self.splits.len();

        // We're ok with an un-bounded channel since the scheduler controls how many cpu tasks
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
            working_set: Default::default(),
            working_set_size: 0,
            target_segment_count: 1000,
            waiting_for_segments: Default::default(),
        }
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

enum ScanTaskResult {
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
                    ScanTaskResult::Filter { .. } => "Filter",
                    ScanTaskResult::Project { .. } => "Project",
                },
            )
            .field(
                "split_idx",
                &match self {
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
    Errored(Arc<VortexError>),
    Finished,
}

pub trait ScanTask: Send {
    /// Execute the scan task on the current thread.
    ///
    /// If this is a projection task, the result array is returned to allow for out-of-order
    /// results for systems that are able to accept them. Otherwise, in-order results are available
    /// from the scheduler.
    ///
    // FIXME(ngates): this cannot return a result, since the worker pools have no way of handling
    //  the error.
    fn execute(&self) -> Option<ArrayRef>;
}

struct FilterTask {
    // TODO(ngates): we may wish to plumb through an is_canceled: Arc<AtomicBool>.
    split_idx: SplitIdx,
    eval: Box<dyn MaskEvaluation>,
    mask: Mask,
    segments: Arc<DashMap<SegmentId, ByteBuffer>>,
    cpu_events: mpsc::UnboundedSender<ScanTaskResult>,
}

impl ScanTask for FilterTask {
    fn execute(&self) -> Option<ArrayRef> {
        let mask = self.eval.invoke(self.mask.clone(), self.segments.as_ref());
        info!("Posting back filter result");
        if let Err(e) = self.cpu_events.unbounded_send(ScanTaskResult::Filter {
            split_idx: self.split_idx,
            mask,
        }) {
            debug!("Failed to send scan task result, scan terminated: {}", e);
        }
        None
    }
}

struct ProjectTask {
    split_idx: SplitIdx,
    eval: Box<dyn ArrayEvaluation>,
    mask: Mask,
    segments: Arc<DashMap<SegmentId, ByteBuffer>>,
    cpu_events: mpsc::UnboundedSender<ScanTaskResult>,
}

impl ScanTask for ProjectTask {
    fn execute(&self) -> Option<ArrayRef> {
        let array = self.eval.invoke(self.mask.clone(), self.segments.as_ref());

        // We take a (zero-)copy of the array for scan drivers that are able to immediately return
        // out-of-order results.
        let result = array.as_ref().ok().cloned();

        info!("Posting back project result");
        if let Err(e) = self.cpu_events.unbounded_send(ScanTaskResult::Project {
            split_idx: self.split_idx,
            array,
        }) {
            debug!("Failed to send scan task result, scan terminated: {}", e);
        }

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
                Ok(event) => {
                    info!("Handling I/O event for segment {:?}", event.segment_id);
                    self.handle_io_event(event)
                }
                Err(e) => {
                    info!("I/O error {:?}", e);
                    self.errored = true;
                    return Poll::Ready(Err(e));
                }
            }
        }

        // Next we handle CPU events.
        while let Poll::Ready(Some(event)) = self.result_recv.poll_next_unpin(cx) {
            info!("Handling CPU event for segment {:?}", event);
            made_progress = true;
            self.handle_cpu_event(event);
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

        // Now we can look to launch new splits based on the total working set size and
        // in-flight request sizes. We don't currently know the in-flight segment sizes, so we
        // will just do it based on segment count instead.
        while self.active_splits.end < self.split_state.len()
            && ((self.working_set.len() + self.segment_futures.len() < self.target_segment_count)
                || self.active_splits.is_empty())
        {
            info!("Launching split {}", self.active_splits.end);
            made_progress = true;
            self.make_progress_on_split(self.active_splits.end);
            self.active_splits.end += 1;
        }
        //
        // // Finally, after making sure we've driven the scheduler, we can emit any buffered array
        // // results. We do it in this order to ensure that we always have in-flight I/O and CPU
        // // work before allowing the caller to process a result.
        // if let Some(array) = self.output_buffer.pop_front() {
        //     return Poll::Ready(Some(array));
        // }

        // Check for termination.
        if self.active_splits.start == self.split_state.len() {
            info!("Completed all splits, terminating");
            self.finished = true;
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
                self.make_progress_on_split(split_idx);
            }
        }
    }

    /// Handles a CPU event by updating the split state based on the result of a filter or
    /// project task.
    ///
    /// Finishes by driving the state machines of any impacted splits.
    fn handle_cpu_event(&mut self, event: ScanTaskResult) {
        info!("Handling CPU event {:?}", event);
        match event {
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
                        info!(
                            "Spawning filter task split: {}, conjunct: {}",
                            split_idx, conjunct_idx
                        );
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
                                waiting_for: self.launch_segment_requests(
                                    split_idx,
                                    Atom::Filter(conjunct_idx + 1),
                                ),
                            })
                        }
                    } else {
                        None
                    }
                }
                SplitState::PendingProject { mask, waiting_for } => {
                    if waiting_for.is_empty() {
                        // If we have all our segments, we can launch the project task.
                        info!("Spawning projection task split: {}", split_idx);
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
                    if let Some(array) = result {
                        self.output_buffer.push_back(Ok(array.clone()));
                        Some(SplitState::Finished)
                    } else {
                        None
                    }
                }
                SplitState::Errored(e) => {
                    self.output_buffer
                        .push_back(Err(VortexError::from(e.clone())));
                    Some(SplitState::Finished)
                }
                SplitState::Finished => {
                    // We're already finished, no progress to make
                    None
                }
            };

            let made_progress = new_state.is_some();
            if let Some(new_state) = &new_state {
                info!(
                    "Moved split {} from {:?} to {:?}",
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
            Atom::Filter(conjunct_idx) => &self.splits[split_idx].filter_segments[conjunct_idx],
            Atom::Project => &self.splits[split_idx].projection_segments,
        };

        for segment_id in segment_ids.iter().copied() {
            if self.working_set.contains_key(&segment_id) {
                continue;
            }
            info!("Requesting segment {:?}", segment_id);
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

impl Stream for Scheduler {
    type Item = VortexResult<ArrayRef>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        loop {
            let pending = match self.make_progress_with_cx(cx) {
                Poll::Ready(Ok(())) => false,
                Poll::Ready(Err(e)) => {
                    return Poll::Ready(Some(Err(e)));
                }
                Poll::Pending => true,
            };

            if let Some(array) = self.output_buffer.pop_front() {
                return Poll::Ready(Some(array));
            }

            if self.finished {
                return Poll::Ready(None);
            }

            if pending {
                return Poll::Pending;
            }
        }
    }
}
