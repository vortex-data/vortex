// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

mod multithread;
mod pool;
mod segments;
#[cfg(feature = "tokio")]
mod tokio;

use bit_vec::BitVec;
use futures::StreamExt;
use futures::channel::mpsc;
use futures::task::noop_waker;
use itertools::Itertools;
use std::collections::VecDeque;
use std::fmt::{Debug, Formatter};
use std::ops::{BitAnd, Range};
use std::sync::Arc;
use std::task::{Context, Poll};
use vortex_array::ArrayRef;
use vortex_error::{VortexExpect, VortexResult};
use vortex_layout::segments::{SegmentId, Segments};
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
    evaluations: Vec<Option<Evaluation>>,
    segments: SegmentCache,
}

pub(crate) trait TaskSpawner: Send {
    fn spawn_task(&self, task: Box<dyn ScanTask>);
}

impl Scan2 {
    pub fn try_new(ranges: Vec<Range<u64>>, ctx: TaskContext<ArrayRef>) -> VortexResult<Self> {
        let nsplits = ranges.len();
        let nconjuncts = ctx.filter.as_ref().map_or(0, |f| f.conjuncts().len());

        let mut splits = Vec::with_capacity(nsplits);
        let mut evaluations = Vec::with_capacity((nsplits * 2 * nconjuncts) + nsplits);
        let mut segments = SegmentCache::new(ctx.segment_source.clone());

        for row_range in ranges.into_iter() {
            // TODO(ngates): we really must clean up this selection logic and push it all inside
            //  the selection object.
            // Step 1: using the caller-provided row range and selection, attempt to disregard this split.
            let row_range = match &ctx.row_range {
                None => row_range,
                Some(scan_row_range) => {
                    if scan_row_range.start >= row_range.end || scan_row_range.end < row_range.start
                    {
                        // No overlap for this task
                        continue;
                    }

                    let intersect_start = scan_row_range.start.max(row_range.start);
                    let intersect_end = scan_row_range.end.min(row_range.end);
                    intersect_start..intersect_end
                }
            };

            let read_mask = ctx.selection.row_mask(&row_range);
            let row_range = read_mask.row_range();
            let mask = read_mask.into_mask();

            if mask.all_false() {
                // The selection has pruned this split.
                continue;
            }

            let split_idx = splits.len();

            let mut pruning_idx = vec![];
            let mut filter_idx = vec![];
            if let Some(filter) = ctx.filter.as_ref() {
                for (conjunct_idx, conjunct) in filter.conjuncts().iter().enumerate() {
                    let prune_eval = ctx.reader.pruning_evaluation(&row_range, conjunct)?;
                    let prune_eval = Evaluation::new_pruning(split_idx, conjunct_idx, prune_eval);
                    pruning_idx.push(evaluations.len());
                    evaluations.push(Some(prune_eval));

                    let filter_eval = ctx.reader.filter_evaluation(&row_range, conjunct)?;
                    let filter_eval = Evaluation::new_filter(split_idx, conjunct_idx, filter_eval);
                    filter_idx.push(evaluations.len());
                    evaluations.push(Some(filter_eval));
                }
            }

            let projection = ctx
                .reader
                .projection_evaluation(&row_range, &ctx.projection)?;
            let projection = Evaluation::new_project(split_idx, projection);
            let projection_idx = evaluations.len();
            evaluations.push(Some(projection));

            let split = Split {
                state: State::InProgress,
                mask,
                pruning: pruning_idx,
                ready_pruning_conjuncts: BitVec::from_elem(nconjuncts, false),
                completed_pruning_conjuncts: BitVec::from_elem(nconjuncts, false),
                filters: filter_idx,
                ready_filter_conjuncts: BitVec::from_elem(nconjuncts, false),
                completed_filter_conjuncts: BitVec::from_elem(nconjuncts, false),
                projection: projection_idx,
                ready_projection: false,
            };

            // segments.acquire(split.all_segments());
            splits.push(split);
        }

        Ok(Self {
            ctx,
            splits,
            evaluations,
            segments,
        })
    }

    pub(crate) fn into_scheduler(self, task_spawner: Box<dyn TaskSpawner>) -> Scheduler {
        // We're ok with an unbounded channel since the scheduler controls how many cpu tasks
        // are in-flight.
        let (cpu_send, cpu_recv) = mpsc::unbounded();

        // Build the inverse map of evaluation segments.
        let mut evaluation_segments: HashMap<SegmentId, Vec<EvaluationIdx>> = HashMap::default();
        for (eval_idx, evaluation) in self.evaluations.iter().flatten().enumerate() {
            for segment_id in &evaluation.waiting_for {
                evaluation_segments
                    .entry(*segment_id)
                    .or_default()
                    .push(eval_idx)
            }
        }

        let nconjuncts = self.ctx.filter.as_ref().map_or(0, |f| f.conjuncts().len());

        Scheduler {
            task_spawner,
            output_buffer: Default::default(),
            result_send: cpu_send,
            result_recv: cpu_recv,
            finished: self.splits.is_empty(),
            errored: false,

            splits: self.splits,
            segments: self.segments,

            completed_splits: 0,

            next_pruning_io_split: 0,
            next_filter_io_splits: vec![0; nconjuncts],
            next_projection_io_split: 0,

            evaluations: self.evaluations,
            evaluation_segments,
            pending_prunings: (0..nconjuncts).map(|_| VecDeque::new()).collect(),
            pending_filters: (0..nconjuncts).map(|_| VecDeque::new()).collect(),
            pending_projections: VecDeque::new(),
            inflight_tasks: 0,

            filter: self.ctx.filter.clone(),
            // TODO(ngates): this should really be target segment size and we control the max
            //  memory consumption.
            target_segment_count: 128,
            target_inflight_tasks: 16,
        }
    }
}

/// Scheduler for a Vortex scan.
///
/// Decides which segments to request and when, as well as spawning CPU work when available.
/// This scheduler is wrapped up for various threading models.
pub(crate) struct Scheduler {
    task_spawner: Box<dyn TaskSpawner>,
    filter: Option<Arc<FilterExpr>>,

    /// A buffer for the arrays emitted by the scan. This buffer is used to return arrays with
    /// a total ordering over the splits of the scan.
    output_buffer: VecDeque<VortexResult<ArrayRef>>,

    /// Results for scan tasks. We allow unbounded channels since the scheduler controls how many
    /// CPU tasks have been spawned.
    result_send: mpsc::UnboundedSender<ScanTaskResult>,
    result_recv: mpsc::UnboundedReceiver<ScanTaskResult>,

    /// If all splits have been processed, we can stop.
    finished: bool,
    /// If any I/O request errors, we want all workers to stop with error.
    errored: bool,

    /// The range of splits that we are currently processing. All splits before should be
    /// finished, and all splits after are pending.
    completed_splits: usize,

    /// The next split for which we will launch I/O.
    next_pruning_io_split: SplitIdx,
    next_filter_io_splits: Vec<SplitIdx>, // By conjunct
    next_projection_io_split: SplitIdx,

    evaluations: Vec<Option<Evaluation>>,
    evaluation_segments: HashMap<SegmentId, Vec<EvaluationIdx>>,

    pending_prunings: Vec<VecDeque<Evaluation>>, // By conjunct.
    pending_filters: Vec<VecDeque<Evaluation>>,  // By conjunct.
    pending_projections: VecDeque<Evaluation>,

    inflight_tasks: usize,
    target_inflight_tasks: usize,

    splits: Vec<Split>,

    // The segment cache used to power the scan.
    segments: SegmentCache,

    /// The target number of segments to hold in memory.
    /// TODO(ngates): use segment size once the segment source reports it.
    target_segment_count: usize,
}

/// Index into the `Scheduler::evaluations` vector.
type EvaluationIdx = usize;

/// Represents the current state of a single evaluation, i.e., pruning, filter, or projection.
struct Evaluation {
    split_idx: SplitIdx,
    eval: Eval,
    waiting_for: HashSet<SegmentId>,
}

impl Debug for Evaluation {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("Evaluation");
        ds.field("split_idx", &self.split_idx);

        match &self.eval {
            Eval::Prune(conjunct, _) => {
                ds.field("eval", &"Prune").field("conjunct", conjunct);
            }
            Eval::Filter(conjunct, _) => {
                ds.field("eval", &"Filter").field("conjunct", conjunct);
            }
            Eval::Project(_) => {
                ds.field("eval", &"Project");
            }
        }

        ds.finish()
    }
}

impl Evaluation {
    fn new_pruning(
        split_idx: SplitIdx,
        conjunct_idx: usize,
        eval: Box<dyn PruningEvaluation>,
    ) -> Self {
        let mut waiting_for = HashSet::new();
        eval.required_segments(&mut waiting_for);
        Self {
            split_idx,
            eval: Eval::Prune(conjunct_idx, eval),
            waiting_for,
        }
    }

    fn new_filter(split_idx: SplitIdx, conjunct_idx: usize, eval: Box<dyn MaskEvaluation>) -> Self {
        let mut waiting_for = HashSet::new();
        eval.required_segments(&mut waiting_for);
        Self {
            split_idx,
            eval: Eval::Filter(conjunct_idx, eval),
            waiting_for,
        }
    }

    fn new_project(split_idx: SplitIdx, eval: Box<dyn ArrayEvaluation>) -> Self {
        let mut waiting_for = HashSet::new();
        eval.required_segments(&mut waiting_for);
        Self {
            split_idx,
            eval: Eval::Project(eval),
            waiting_for,
        }
    }
}

enum Eval {
    Prune(usize, Box<dyn PruningEvaluation>),
    Filter(usize, Box<dyn MaskEvaluation>),
    Project(Box<dyn ArrayEvaluation>),
}

struct Split {
    state: State,
    mask: Mask,

    pruning: Vec<EvaluationIdx>,
    ready_pruning_conjuncts: BitVec,
    completed_pruning_conjuncts: BitVec,

    filters: Vec<EvaluationIdx>,
    ready_filter_conjuncts: BitVec,
    completed_filter_conjuncts: BitVec,

    projection: EvaluationIdx,
    ready_projection: bool,
}

enum State {
    InProgress,
    Pruned,
    Finished(Option<VortexResult<ArrayRef>>),
}

type SplitIdx = usize;

enum ScanTaskResult {
    Prune {
        split_idx: SplitIdx,
        conjunct_idx: usize,
        result: VortexResult<Mask>,
    },
    Filter {
        split_idx: SplitIdx,
        conjunct_idx: usize,
        result: VortexResult<Mask>,
    },
    Project {
        split_idx: SplitIdx,
        result: VortexResult<ArrayRef>,
    },
}

impl Debug for ScanTaskResult {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let mut ds = f.debug_struct("ScanTaskResult");

        ds.field(
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
        );

        match self {
            ScanTaskResult::Prune { conjunct_idx, .. } => {
                ds.field("conjunct_idx", conjunct_idx);
            }
            ScanTaskResult::Filter { conjunct_idx, .. } => {
                ds.field("conjunct_idx", conjunct_idx);
            }
            ScanTaskResult::Project { .. } => {}
        }

        ds.finish()
    }
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
    conjunct_idx: usize,
    eval: Box<dyn PruningEvaluation>,
    mask: Mask,
    segments: Arc<dyn Segments>,
    cpu_events: mpsc::UnboundedSender<ScanTaskResult>,
}

impl ScanTask for PruneTask {
    fn execute(&self) -> Option<ArrayRef> {
        let result = self.eval.invoke(self.mask.clone(), self.segments.as_ref());
        // Ignore the error, since it means the scan has terminated early.
        let _ = self.cpu_events.unbounded_send(ScanTaskResult::Prune {
            split_idx: self.split_idx,
            conjunct_idx: self.conjunct_idx,
            result,
        });
        None
    }
}

struct FilterTask {
    // TODO(ngates): we may wish to plumb through an is_canceled: Arc<AtomicBool>.
    split_idx: SplitIdx,
    conjunct_idx: usize,
    eval: Box<dyn MaskEvaluation>,
    mask: Mask,
    segments: Arc<dyn Segments>,
    cpu_events: mpsc::UnboundedSender<ScanTaskResult>,
}

impl ScanTask for FilterTask {
    fn execute(&self) -> Option<ArrayRef> {
        let result = self.eval.invoke(self.mask.clone(), self.segments.as_ref());
        // Ignore the error, since it means the scan has terminated early.
        let _ = self.cpu_events.unbounded_send(ScanTaskResult::Filter {
            split_idx: self.split_idx,
            conjunct_idx: self.conjunct_idx,
            result,
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
        let result = self.eval.invoke(self.mask.clone(), self.segments.as_ref());

        // We take a (zero-)copy of the array for scan drivers that are able to immediately return
        // out-of-order results.
        let array = result.as_ref().ok().cloned();

        // Ignore the error, since it means the scan has terminated early.
        let _ = self.cpu_events.unbounded_send(ScanTaskResult::Project {
            split_idx: self.split_idx,
            result,
        });
        array
    }
}

impl Scheduler {
    fn make_progress(&mut self) -> Poll<VortexResult<()>> {
        let waker = noop_waker();
        let mut ctx = Context::from_waker(&waker);
        self.make_progress_with_cx(&mut ctx)
    }

    fn make_progress_with_cx(&mut self, cx: &mut Context) -> Poll<VortexResult<()>> {
        let mut made_progress = false;

        // 1. handle I/O events
        while let Poll::Ready(Some(result)) = self.segments.poll_next_unpin(cx) {
            made_progress = true;
            match result {
                Ok(segment_id) => self.handle_io_event(segment_id),
                Err(e) => {
                    self.errored = true;
                    return Poll::Ready(Err(e));
                }
            }
        }

        // 2. handle CPU events.
        while let Poll::Ready(Some(event)) = self.result_recv.poll_next_unpin(cx) {
            made_progress = true;
            self.inflight_tasks -= 1;
            self.handle_cpu_event(event);
        }

        // 3. Mark as complete any splits that have finished.
        while self.completed_splits < self.splits.len() {
            match &mut self.splits[self.completed_splits].state {
                State::Finished(result) => {
                    if let Some(result) = result.take() {
                        log::debug!("Split {} finished", self.completed_splits);
                        self.output_buffer.push_back(result);
                    }
                    self.completed_splits += 1;
                }
                State::Pruned => {
                    log::debug!("Split {} pruned", self.completed_splits);
                    self.completed_splits += 1;
                }
                _ => break,
            }
        }
        // Mark ourselves as done.
        if self.completed_splits == self.splits.len() {
            self.finished = true;
        }

        // 4. Spawn any pending CPU tasks.
        'cpu: while self.inflight_tasks < self.target_inflight_tasks {
            // The top priority is projection tasks, since they can actually emit data.
            if let Some(eval) = self.pending_projections.pop_front() {
                let mask = self.splits[eval.split_idx].mask.clone();
                log::debug!("Spawning projection task for split {}", eval.split_idx);
                self.spawn_task(eval, mask);
                made_progress = true;
                self.inflight_tasks += 1;
                continue 'cpu;
            }

            // We look across each of the pending task queues and launch work based on some
            // priority function. Each pending queue is internally sorted by the row offset.
            // FIXME(ngates): we need to make that true.
            for pending in &mut self.pending_prunings {
                if let Some(eval) = pending.pop_front() {
                    let mask = self.splits[eval.split_idx].mask.clone();
                    log::debug!("Spawning pruning task for split {}", eval.split_idx);
                    self.spawn_task(eval, mask);
                    made_progress = true;
                    self.inflight_tasks += 1;
                    continue 'cpu;
                }
            }

            // Now we spawn tasks from each conjunct
            // FIXME(ngates): the order of this is weird.
            for filter in &mut self.pending_filters {
                if let Some(eval) = filter.pop_front() {
                    let mask = self.splits[eval.split_idx].mask.clone();
                    log::debug!("Spawning filter task for split {}", eval.split_idx);
                    self.spawn_task(eval, mask);
                    made_progress = true;
                    self.inflight_tasks += 1;
                    continue 'cpu;
                }
            }

            break;
        }

        // 5. Spawn I/O tasks.
        'io: while self.segments.inflight_count() < self.target_segment_count {
            // TODO(ngates): we try to make sure the number of in-flight requests (ideally the
            //  batched count) is reasonable. Although if we get here and we have a big back-log
            //  of CPU work, then we know we're CPU bound and not I/O bound and perhaps we should
            //  hold off on I/O for a bit.

            // 1. We try to spawn I/O for pruning.
            if self.next_pruning_io_split < self.splits.len() {
                let split_idx = self.next_pruning_io_split;
                let split = &self.splits[split_idx];
                log::debug!("Requesting segments for pruning split {}", split_idx);
                for eval_idx in &split.pruning {
                    if let Some(eval) = &self.evaluations[*eval_idx] {
                        made_progress = true;
                        self.segments.request(&eval.waiting_for);
                    }
                }
                self.next_pruning_io_split += 1;
                continue 'io;
            }

            // 2. We try to spawn I/O based on conjunct selectivity?
            if let Some(filter) = self.filter.as_ref() {
                let filter_run_ahead = 16; // Run ahead 16 splits.

                // TODO(ngates): for now, we just launch all filter I/O.
                for conjunct_idx in 0..filter.conjuncts().len() {
                    if self.next_filter_io_splits[conjunct_idx] < self.splits.len() {
                        let split_idx = self.next_filter_io_splits[conjunct_idx];

                        // Only keep going if we haven't reached the end of the filter run ahead.
                        if split_idx <= self.next_projection_io_split + filter_run_ahead {
                            let split = &self.splits[split_idx];
                            let eval_idx = split.filters[conjunct_idx];
                            if let Some(eval) = &self.evaluations[eval_idx] {
                                made_progress = true;
                                self.segments.request(&eval.waiting_for);
                            }
                            self.next_filter_io_splits[conjunct_idx] += 1;
                            continue 'io;
                        }
                    }
                }
            }

            // 3. Try to spawn I/O for projection.
            if self.next_projection_io_split < self.splits.len() {
                let split_idx = self.next_projection_io_split;
                let split = &self.splits[split_idx];
                if let Some(eval) = &self.evaluations[split.projection] {
                    made_progress = true;
                    self.segments.request(&eval.waiting_for);
                }
                self.next_projection_io_split += 1;
                continue 'io;
            }

            // 4. If we got here, then we've finished the loop for this iteration.
            break;
        }

        if made_progress {
            Poll::Ready(Ok(()))
        } else {
            Poll::Pending
        }
    }

    /// Handles an I/O event by updating the splits that are waiting for this particular segment.
    /// Finishes by driving the state machines of any impacted splits.
    fn handle_io_event(&mut self, segment_id: SegmentId) {
        // Check which evaluations are waiting for this segment.
        log::debug!("Handle IO event: {:?}", segment_id);
        if let Some(eval_idxs) = self.evaluation_segments.remove(&segment_id) {
            for eval_idx in eval_idxs {
                let eval = self.evaluations[eval_idx].as_mut().vortex_expect(
                    "Evaluation cannot have been consumed since it depends on this segment",
                );
                eval.waiting_for.remove(&segment_id);

                if eval.waiting_for.is_empty() {
                    let split = &mut self.splits[eval.split_idx];
                    match eval.eval {
                        Eval::Prune(conjunct_idx, _) => {
                            split.ready_pruning_conjuncts.set(conjunct_idx, true)
                        }
                        Eval::Filter(conjunct_idx, _) => {
                            split.ready_filter_conjuncts.set(conjunct_idx, true)
                        }
                        Eval::Project(_) => split.ready_projection = true,
                    }

                    let split_idx = eval.split_idx;
                    self.make_progress_on_split(split_idx);
                }
            }
        }
    }

    /// Handles a CPU event by updating the split state based on the result of a filter or
    /// project task.
    ///
    /// Finishes by driving the state machines of any impacted splits.
    fn handle_cpu_event(&mut self, event: ScanTaskResult) {
        log::debug!("Handle CPU event: {:?}", event);
        match event {
            ScanTaskResult::Prune {
                split_idx,
                conjunct_idx,
                result,
            } => {
                let split = &mut self.splits[split_idx];

                match result {
                    Ok(result) => {
                        split.mask = result.bitand(&split.mask);
                        assert!(!split.completed_pruning_conjuncts[conjunct_idx]);
                        split.completed_pruning_conjuncts.set(conjunct_idx, true);
                    }
                    Err(e) => {
                        // FIXME(ngates): drop the remaining evaluations.
                        split.state = State::Finished(Some(Err(e)));
                        self.errored = true;
                    }
                }

                self.make_progress_on_split(split_idx);
            }
            ScanTaskResult::Filter {
                split_idx,
                conjunct_idx,
                result,
            } => {
                let split = &mut self.splits[split_idx];

                match result {
                    Ok(result) => {
                        split.mask = result.bitand(&split.mask);
                        assert!(!split.completed_filter_conjuncts[conjunct_idx]);
                        split.completed_filter_conjuncts.set(conjunct_idx, true);
                    }
                    Err(e) => {
                        // FIXME(ngates): drop the remaining evaluations.
                        split.state = State::Finished(Some(Err(e)));
                        self.errored = true;
                    }
                }

                self.make_progress_on_split(split_idx);
            }
            ScanTaskResult::Project { split_idx, result } => {
                let split = &mut self.splits[split_idx];

                match result {
                    Ok(result) => {
                        split.state = State::Finished(Some(Ok(result)));
                    }
                    Err(e) => {
                        // FIXME(ngates): drop the remaining evaluations.
                        split.state = State::Finished(Some(Err(e)));
                        self.errored = true;
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
        let mut split = &mut self.splits[split_idx];
        let nconjuncts = split.filters.len();
        match &split.state {
            State::InProgress => {
                // If the mask is all false, we can prune the segment.
                if split.mask.all_false() {
                    split.state = State::Pruned;
                    // self.segments.release(split.remaining_segments().iter());
                    // split.drop_evaluations();
                    return;
                }

                // Attempt to launch any ready pruning tasks.
                for idx in (0..nconjuncts) {
                    if split.ready_pruning_conjuncts[idx] {
                        self.pending_prunings[idx].push_back(
                            self.evaluations[split.pruning[idx]]
                                .take()
                                .vortex_expect("pruning evaluation already taken"),
                        );
                        split.ready_pruning_conjuncts.set(idx, false);
                    }
                }

                // Also, launch as many filters as we can.
                if let Some(filter) = self.filter.as_ref() {
                    while let Some(idx) = filter.next_conjunct(&split.ready_filter_conjuncts) {
                        self.pending_filters[idx].push_back(
                            self.evaluations[split.filters[idx]]
                                .take()
                                .vortex_expect("filter evaluation already taken"),
                        );
                        split.ready_filter_conjuncts.set(idx, false);
                    }
                }

                // If we've completed all filtering, then proceed to projection.
                let completed_pruning = split.completed_pruning_conjuncts.count_ones();
                let completed_filter = split.completed_filter_conjuncts.count_ones();
                if completed_pruning == nconjuncts as u64
                    && completed_filter == nconjuncts as u64
                    && split.ready_projection
                {
                    self.pending_projections.push_back(
                        self.evaluations[split.projection]
                            .take()
                            .vortex_expect("projection evaluation already taken"),
                    );
                    split.ready_projection = false;
                }
            }
            State::Pruned => {}
            State::Finished(_) => {}
        }
    }

    fn spawn_task(&mut self, evaluation: Evaluation, mask: Mask) {
        let task: Box<dyn ScanTask> = match evaluation.eval {
            Eval::Prune(conjunct_idx, eval) => Box::new(PruneTask {
                split_idx: evaluation.split_idx,
                conjunct_idx,
                eval,
                mask,
                segments: self.segments.segments(),
                cpu_events: self.result_send.clone(),
            }),
            Eval::Filter(conjunct_idx, eval) => Box::new(FilterTask {
                split_idx: evaluation.split_idx,
                conjunct_idx,
                eval,
                mask,
                segments: self.segments.segments(),
                cpu_events: self.result_send.clone(),
            }),
            Eval::Project(eval) => Box::new(ProjectTask {
                split_idx: evaluation.split_idx,
                eval,
                mask,
                segments: self.segments.segments(),
                cpu_events: self.result_send.clone(),
            }),
        };

        self.task_spawner.spawn_task(task);
    }
}
