// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scheduler-visible scan tasks.
//!
//! A scan task is a morsel-level runtime unit with explicit read dependencies.
//! Layouts and scan-plan adapters still decide what a task means, while the
//! scheduler can reason about phase, read bytes, and deduplication before the
//! task future is launched.

use std::collections::BTreeMap;
use std::collections::VecDeque;

use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_utils::aliases::hash_map::Entry;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::read::ReadRequestKey;
use crate::read::ReadResults;
use crate::read::ScanIoPhase;
use crate::read::ScanRead;

/// Fine-grained scheduling lane for a scan task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ScanTaskLane {
    /// Scan-domain metadata/index evidence shared by all morsels.
    ScanEvidence {
        /// Predicate slot this evidence task belongs to.
        predicate_idx: u32,
        /// Evidence-provider slot within the predicate.
        evidence_idx: u32,
    },
    /// Metadata/index evidence for one predicate.
    Evidence {
        /// Predicate slot this evidence task belongs to.
        predicate_idx: u32,
        /// Evidence-provider slot within the predicate.
        evidence_idx: u32,
    },
    /// Exact residual evaluation for one predicate.
    Predicate {
        /// Predicate slot this residual-read task belongs to.
        predicate_idx: u32,
    },
    /// Final projected data read.
    Projection,
    /// Aggregate read.
    Aggregate,
}

impl ScanTaskLane {
    /// Default lane for callers that only know the high-level phase.
    pub fn from_phase(phase: ScanIoPhase) -> Self {
        match phase {
            ScanIoPhase::EvidenceProbe | ScanIoPhase::EvidenceSetup => Self::Evidence {
                predicate_idx: 0,
                evidence_idx: 0,
            },
            ScanIoPhase::PredicateRead => Self::Predicate { predicate_idx: 0 },
            ScanIoPhase::ProjectionRead => Self::Projection,
            ScanIoPhase::AggregateRead => Self::Aggregate,
        }
    }

    fn group(self) -> ScanTaskGroup {
        match self {
            Self::ScanEvidence { .. } => ScanTaskGroup::Evidence,
            Self::Evidence { .. } => ScanTaskGroup::Evidence,
            Self::Predicate { .. } => ScanTaskGroup::Predicate,
            Self::Projection | Self::Aggregate => ScanTaskGroup::Projection,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ScanTaskGroup {
    Predicate,
    Projection,
    Evidence,
}

impl ScanTaskGroup {
    fn idx(self) -> usize {
        match self {
            Self::Predicate => 0,
            Self::Projection => 1,
            Self::Evidence => 2,
        }
    }
}

/// Scheduler-visible read dependency for one scan task.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ScanTaskRead {
    /// Dedupe key for the logical read.
    pub key: ReadRequestKey,
    /// Number of bytes this read contributes if it is not already active.
    pub bytes: u64,
}

impl ScanTaskRead {
    /// Convert registered segment reads into scheduler-visible task reads.
    pub fn from_scan_reads(reads: &[ScanRead]) -> Vec<Self> {
        let mut seen = HashSet::new();
        reads
            .iter()
            .filter_map(|read| {
                let key = read.request.key;
                seen.insert(key).then_some(Self {
                    key,
                    bytes: read.request.bytes,
                })
            })
            .collect()
    }
}

/// Reads requested by one scheduler-visible scan step.
#[derive(Default)]
pub struct ScanStepReads {
    required: Vec<ScanRead>,
    prefetch: Vec<ScanRead>,
}

impl ScanStepReads {
    /// Create an empty read set.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a read that must complete before the task can make progress.
    pub fn require(&mut self, read: ScanRead) {
        self.required.push(read);
    }

    /// Add a read that may be fetched speculatively but must not gate task progress.
    pub fn prefetch(&mut self, read: ScanRead) {
        self.prefetch.push(read);
    }

    /// Return required reads.
    pub fn required(&self) -> &[ScanRead] {
        &self.required
    }

    /// Return prefetch reads.
    pub fn prefetches(&self) -> &[ScanRead] {
        &self.prefetch
    }

    /// Consume required reads.
    pub fn into_required(self) -> Vec<ScanRead> {
        self.required
    }

    /// Consume prefetch reads.
    pub fn into_prefetches(self) -> Vec<ScanRead> {
        self.prefetch
    }

    /// Consume both read classes.
    pub fn into_parts(self) -> (Vec<ScanRead>, Vec<ScanRead>) {
        (self.required, self.prefetch)
    }

    /// Return true when there are no reads of either class.
    pub fn is_empty(&self) -> bool {
        self.required.is_empty() && self.prefetch.is_empty()
    }

    /// Return true when no progress-gating reads were requested.
    pub fn required_is_empty(&self) -> bool {
        self.required.is_empty()
    }
}

/// Result of executing a scheduler-visible scan step.
pub enum ScanStepResult<T> {
    /// The task produced its final output.
    Ready(T),
    /// The task needs another scheduler-admitted step before it can finish.
    Continue(ScanTaskBox<T>),
}

type ScanStepContinuation<T> =
    Box<dyn FnOnce(ReadResults) -> VortexResult<ScanStepResult<T>> + Send>;

/// One scheduler-visible step of a morsel-level scan task.
pub struct ScanStep<T> {
    morsel_id: usize,
    phase: ScanIoPhase,
    lane: ScanTaskLane,
    reads: Vec<ScanTaskRead>,
    /// Reads that must resolve before the continuation runs.
    pub required_reads: Vec<ScanRead>,
    /// Reads that may be fetched speculatively while this step is queued.
    pub prefetch_reads: Vec<ScanRead>,
    priority: u64,
    continuation: Option<ScanStepContinuation<T>>,
}

impl<T: Send + 'static> ScanStep<T> {
    /// Default scheduling priority for tasks without a more specific estimate.
    pub const DEFAULT_PRIORITY: u64 = 1_000_000;

    /// Create a scheduler-visible scan step.
    pub fn new(
        morsel_id: usize,
        phase: ScanIoPhase,
        lane: ScanTaskLane,
        reads: Vec<ScanTaskRead>,
        required_reads: Vec<ScanRead>,
        prefetch_reads: Vec<ScanRead>,
        continuation: impl FnOnce(ReadResults) -> VortexResult<ScanStepResult<T>> + Send + 'static,
    ) -> Self {
        Self {
            morsel_id,
            phase,
            lane,
            reads,
            required_reads,
            prefetch_reads,
            priority: Self::DEFAULT_PRIORITY,
            continuation: Some(Box::new(continuation)),
        }
    }

    /// Create a ready step with no required reads.
    pub fn ready(
        morsel_id: usize,
        phase: ScanIoPhase,
        lane: ScanTaskLane,
        reads: Vec<ScanTaskRead>,
        output: VortexResult<T>,
    ) -> Self
    where
        T: Send + 'static,
    {
        Self::new(
            morsel_id,
            phase,
            lane,
            reads,
            Vec::new(),
            Vec::new(),
            move |_| output.map(ScanStepResult::Ready),
        )
    }

    /// Return this step with an explicit scheduling priority.
    pub fn with_priority(mut self, priority: u64) -> Self {
        self.priority = priority;
        self
    }

    /// Box this step behind the [`ScanTask`] trait.
    pub fn boxed(self) -> ScanTaskBox<T>
    where
        T: 'static,
    {
        Box::new(self)
    }

    /// Reads that must resolve before the continuation runs.
    pub fn required_reads(&self) -> &[ScanRead] {
        &self.required_reads
    }

    /// Reads that may be fetched speculatively for this step.
    pub fn prefetch_reads(&self) -> &[ScanRead] {
        &self.prefetch_reads
    }

    /// Consume the step into its required and prefetch reads.
    pub fn into_reads(self) -> (Vec<ScanRead>, Vec<ScanRead>) {
        (self.required_reads, self.prefetch_reads)
    }

    /// Take the step's required and prefetch reads, leaving empty read lists behind.
    pub fn take_reads(&mut self) -> (Vec<ScanRead>, Vec<ScanRead>) {
        (
            std::mem::take(&mut self.required_reads),
            std::mem::take(&mut self.prefetch_reads),
        )
    }

    /// Execute this step's continuation.
    pub fn continue_with(mut self, results: ReadResults) -> VortexResult<ScanStepResult<T>> {
        let continuation = self
            .continuation
            .take()
            .ok_or_else(|| vortex_err!("scan step was continued after completion"))?;
        continuation(results)
    }
}

/// A morsel-level scan task with explicit read dependencies.
pub trait ScanTask<T>: Send {
    /// Morsel identifier this task belongs to.
    fn morsel_id(&self) -> usize;

    /// High-level scan phase for scheduling.
    fn phase(&self) -> ScanIoPhase;

    /// Fine-grained scheduling lane.
    fn lane(&self) -> ScanTaskLane;

    /// Logical reads required by this task.
    fn reads(&self) -> &[ScanTaskRead];

    /// Scheduling priority within this task's group. Lower values run first.
    fn priority(&self) -> u64;

    /// Convert this task into its next scheduler-visible step.
    fn into_step(self: Box<Self>) -> VortexResult<ScanStep<T>>;
}

/// Boxed scan task.
pub type ScanTaskBox<T> = Box<dyn ScanTask<T>>;

impl<T: Send + 'static> ScanTask<T> for ScanStep<T> {
    fn morsel_id(&self) -> usize {
        self.morsel_id
    }

    fn phase(&self) -> ScanIoPhase {
        self.phase
    }

    fn lane(&self) -> ScanTaskLane {
        self.lane
    }

    fn reads(&self) -> &[ScanTaskRead] {
        &self.reads
    }

    fn priority(&self) -> u64 {
        self.priority
    }

    fn into_step(self: Box<Self>) -> VortexResult<ScanStep<T>> {
        Ok(*self)
    }
}

/// A task admitted for launch, including the reads that contributed to the active byte budget.
pub struct AdmittedScanTask<T> {
    task: ScanTaskBox<T>,
    lane: ScanTaskLane,
    admitted_reads: Vec<ScanTaskRead>,
}

impl<T> AdmittedScanTask<T> {
    /// Create an admitted task.
    fn new(task: ScanTaskBox<T>, admitted_reads: Vec<ScanTaskRead>) -> Self {
        let lane = task.lane();
        Self {
            task,
            lane,
            admitted_reads,
        }
    }

    /// Scheduling lane for this launched task.
    pub fn lane(&self) -> ScanTaskLane {
        self.lane
    }

    /// Borrow reads admitted for this task.
    pub fn admitted_reads(&self) -> &[ScanTaskRead] {
        &self.admitted_reads
    }

    /// Consume this value into the task and admitted reads.
    pub fn into_parts(self) -> (ScanTaskBox<T>, ScanTaskLane, Vec<ScanTaskRead>) {
        (self.task, self.lane, self.admitted_reads)
    }
}

#[derive(Clone, Copy, Debug)]
struct ActiveRead {
    bytes: u64,
    refs: usize,
}

/// Queue of scheduler-visible scan tasks with byte-budgeted read admission.
pub struct ScanTaskQueue<T> {
    evidence_queues: BTreeMap<(u32, u32), VecDeque<ScanTaskBox<T>>>,
    predicate_queues: BTreeMap<u32, VecDeque<ScanTaskBox<T>>>,
    projection_queue: VecDeque<ScanTaskBox<T>>,
    read_byte_budget: u64,
    active_read_bytes: u64,
    active_group_read_bytes: [u64; 3],
    active_reads: HashMap<ReadRequestKey, ActiveRead>,
}

impl<T> ScanTaskQueue<T> {
    /// Create an empty task queue with an in-flight logical read-byte budget.
    pub fn new(read_byte_budget: u64) -> Self {
        Self {
            evidence_queues: BTreeMap::new(),
            predicate_queues: BTreeMap::new(),
            projection_queue: VecDeque::new(),
            read_byte_budget,
            active_read_bytes: 0,
            active_group_read_bytes: [0; 3],
            active_reads: HashMap::new(),
        }
    }

    /// Push a task into the queue for its phase.
    pub fn push(&mut self, task: ScanTaskBox<T>) {
        let phase = task.phase();
        let lane = task.lane();
        tracing::trace!(
            target: "vortex_scan::task",
            morsel_id = task.morsel_id(),
            ?phase,
            ?lane,
            read_count = task.reads().len(),
            read_bytes = scan_task_read_bytes(task.reads()),
            "queued scan task"
        );
        match lane {
            ScanTaskLane::ScanEvidence {
                predicate_idx,
                evidence_idx,
            }
            | ScanTaskLane::Evidence {
                predicate_idx,
                evidence_idx,
            } => self
                .evidence_queues
                .entry((predicate_idx, evidence_idx))
                .or_default()
                .push_back(task),
            ScanTaskLane::Predicate { predicate_idx } => self
                .predicate_queues
                .entry(predicate_idx)
                .or_default()
                .push_back(task),
            ScanTaskLane::Projection | ScanTaskLane::Aggregate => {
                self.projection_queue.push_back(task)
            }
        }
    }

    /// Extend this queue with tasks.
    pub fn extend(&mut self, tasks: impl IntoIterator<Item = ScanTaskBox<T>>) {
        for task in tasks {
            self.push(task);
        }
    }

    /// Return whether no tasks are queued.
    pub fn is_empty(&self) -> bool {
        self.evidence_queues.values().all(VecDeque::is_empty)
            && self.predicate_queues.values().all(VecDeque::is_empty)
            && self.projection_queue.is_empty()
    }

    /// Clear queued tasks and active read accounting.
    pub fn clear(&mut self) {
        self.evidence_queues.clear();
        self.predicate_queues.clear();
        self.projection_queue.clear();
        self.active_read_bytes = 0;
        self.active_group_read_bytes = [0; 3];
        self.active_reads.clear();
    }

    /// Number of queued evidence tasks.
    pub fn evidence_len(&self) -> usize {
        self.evidence_queues.values().map(VecDeque::len).sum()
    }

    /// Number of queued predicate-read tasks.
    pub fn predicate_len(&self) -> usize {
        self.predicate_queues.values().map(VecDeque::len).sum()
    }

    /// Number of queued projection-read tasks.
    pub fn projection_len(&self) -> usize {
        self.projection_queue.len()
    }

    /// Number of currently active logical read bytes.
    pub fn active_read_bytes(&self) -> u64 {
        self.active_read_bytes
    }

    /// Number of active logical read dependencies.
    pub fn active_read_count(&self) -> usize {
        self.active_reads.len()
    }

    /// Number of currently active predicate logical read bytes.
    pub fn active_predicate_read_bytes(&self) -> u64 {
        self.active_group_read_bytes[ScanTaskGroup::Predicate.idx()]
    }

    /// Number of currently active projection logical read bytes.
    pub fn active_projection_read_bytes(&self) -> u64 {
        self.active_group_read_bytes[ScanTaskGroup::Projection.idx()]
    }

    /// Number of currently active evidence logical read bytes.
    pub fn active_evidence_read_bytes(&self) -> u64 {
        self.active_group_read_bytes[ScanTaskGroup::Evidence.idx()]
    }

    /// Pop the next task admitted by the active read byte strategy.
    pub fn pop_next_admissible(
        &mut self,
        in_flight_empty: bool,
        mut is_live_morsel: impl FnMut(usize) -> bool,
    ) -> Option<AdmittedScanTask<T>> {
        self.pop_next_admissible_with_projection_gate(in_flight_empty, true, &mut is_live_morsel)
    }

    /// Pop the next task admitted by the active read byte strategy, optionally suppressing
    /// projection/aggregate work.
    ///
    /// This is useful when a caller wants predicate/evidence run-ahead but must avoid producing
    /// more output batches until downstream has consumed earlier projection results.
    pub fn pop_next_admissible_with_projection_gate(
        &mut self,
        in_flight_empty: bool,
        projection_admissible: bool,
        mut is_live_morsel: impl FnMut(usize) -> bool,
    ) -> Option<AdmittedScanTask<T>> {
        self.drop_dead_heads(&mut is_live_morsel);

        for (group, enforce_target) in [
            (ScanTaskGroup::Evidence, true),
            (ScanTaskGroup::Predicate, true),
            (ScanTaskGroup::Projection, true),
            (ScanTaskGroup::Predicate, false),
            (ScanTaskGroup::Projection, false),
            (ScanTaskGroup::Evidence, false),
        ] {
            if group == ScanTaskGroup::Projection && !projection_admissible {
                continue;
            }
            if let Some(task) = self.pop_group_admissible(group, enforce_target, in_flight_empty) {
                return Some(task);
            }
        }

        None
    }

    fn pop_group_admissible(
        &mut self,
        group: ScanTaskGroup,
        enforce_target: bool,
        in_flight_empty: bool,
    ) -> Option<AdmittedScanTask<T>> {
        if enforce_target && !self.group_has_budget(group, 0, in_flight_empty) {
            return None;
        }

        let active_reads = &self.active_reads;
        let active_read_bytes = self.active_read_bytes;
        let read_byte_budget = self.read_byte_budget;

        match group {
            ScanTaskGroup::Predicate => {
                let mut best = None;
                for (idx, queue) in &self.predicate_queues {
                    let Some(task) = queue.front() else {
                        continue;
                    };
                    let score = TaskScore::new(
                        active_reads,
                        task.reads(),
                        task.priority(),
                        task.morsel_id(),
                        *idx,
                    );
                    if !can_admit_task(
                        active_read_bytes,
                        read_byte_budget,
                        in_flight_empty,
                        score.incremental_read_bytes,
                    ) || (enforce_target
                        && !self.group_has_budget(group, score.read_bytes, in_flight_empty))
                    {
                        continue;
                    }
                    if best.is_none_or(|(_, best_score)| score < best_score) {
                        best = Some((*idx, score));
                    }
                }
                let (idx, _) = best?;
                let task = self.predicate_queues.get_mut(&idx)?.pop_front()?;
                Some(self.admit_task(task))
            }
            ScanTaskGroup::Projection => {
                let task = self.projection_queue.front()?;
                let score = TaskScore::new(
                    active_reads,
                    task.reads(),
                    task.priority(),
                    task.morsel_id(),
                    0,
                );
                if !can_admit_task(
                    active_read_bytes,
                    read_byte_budget,
                    in_flight_empty,
                    score.incremental_read_bytes,
                ) || (enforce_target
                    && !self.group_has_budget(group, score.read_bytes, in_flight_empty))
                {
                    return None;
                }
                let task = self.projection_queue.pop_front()?;
                Some(self.admit_task(task))
            }
            ScanTaskGroup::Evidence => {
                let mut best = None;
                for (idx, queue) in &self.evidence_queues {
                    let Some(task) = queue.front() else {
                        continue;
                    };
                    let score = TaskScore::new(
                        active_reads,
                        task.reads(),
                        task.priority(),
                        task.morsel_id(),
                        idx.0,
                    );
                    if !can_admit_task(
                        active_read_bytes,
                        read_byte_budget,
                        in_flight_empty,
                        score.incremental_read_bytes,
                    ) || (enforce_target
                        && !self.group_has_budget(group, score.read_bytes, in_flight_empty))
                    {
                        continue;
                    }
                    if best.is_none_or(|(_, best_score)| score < best_score) {
                        best = Some((*idx, score));
                    }
                }
                let (idx, _) = best?;
                let task = self.evidence_queues.get_mut(&idx)?.pop_front()?;
                Some(self.admit_task(task))
            }
        }
    }

    fn drop_dead_heads(&mut self, is_live_morsel: &mut impl FnMut(usize) -> bool) {
        drop_dead_heads_from_map(&mut self.evidence_queues, &mut |task| {
            !matches!(task.lane(), ScanTaskLane::ScanEvidence { .. })
                && !is_live_morsel(task.morsel_id())
        });
        drop_dead_heads_from_map(&mut self.predicate_queues, &mut |task| {
            !is_live_morsel(task.morsel_id())
        });
        while self
            .projection_queue
            .front()
            .is_some_and(|task| !is_live_morsel(task.morsel_id()))
        {
            self.projection_queue.pop_front();
        }
    }

    fn group_target_bytes(&self, group: ScanTaskGroup) -> u64 {
        if self.read_byte_budget == u64::MAX {
            return u64::MAX;
        }

        let projection = (self.read_byte_budget / 8).max(1);
        let evidence = (self.read_byte_budget / 8).max(1);
        match group {
            ScanTaskGroup::Predicate => self
                .read_byte_budget
                .saturating_sub(projection)
                .saturating_sub(evidence)
                .max(1),
            ScanTaskGroup::Projection => projection,
            ScanTaskGroup::Evidence => evidence,
        }
    }

    fn group_has_budget(
        &self,
        group: ScanTaskGroup,
        task_read_bytes: u64,
        in_flight_empty: bool,
    ) -> bool {
        let active = self.active_group_read_bytes[group.idx()];
        let target = self.group_target_bytes(group);
        active < target || active.saturating_add(task_read_bytes) <= target || in_flight_empty
    }

    fn admit_task(&mut self, task: ScanTaskBox<T>) -> AdmittedScanTask<T> {
        let phase = task.phase();
        let lane = task.lane();
        let group = lane.group();
        let morsel_id = task.morsel_id();
        let read_count = task.reads().len();
        let read_bytes = scan_task_read_bytes(task.reads());
        let incremental_read_bytes = incremental_read_bytes(&self.active_reads, task.reads());
        let mut admitted = Vec::with_capacity(task.reads().len());
        let mut seen = HashSet::new();
        for read in task.reads() {
            if !seen.insert(read.key) {
                continue;
            }
            match self.active_reads.entry(read.key) {
                Entry::Occupied(mut entry) => {
                    let active = entry.get_mut();
                    active.refs = active.refs.saturating_add(1);
                }
                Entry::Vacant(entry) => {
                    self.active_read_bytes = self.active_read_bytes.saturating_add(read.bytes);
                    entry.insert(ActiveRead {
                        bytes: read.bytes,
                        refs: 1,
                    });
                }
            }
            admitted.push(*read);
        }
        self.active_group_read_bytes[group.idx()] = self.active_group_read_bytes[group.idx()]
            .saturating_add(scan_task_read_bytes(&admitted));
        tracing::trace!(
            target: "vortex_scan::task",
            morsel_id,
            ?phase,
            ?lane,
            read_count,
            read_bytes,
            incremental_read_bytes,
            active_read_bytes = self.active_read_bytes,
            "admitted scan task"
        );
        AdmittedScanTask::new(task, admitted)
    }

    /// Release reads admitted for a completed launched task.
    ///
    /// Returns the logical read keys whose active reference count reached zero.
    pub fn release_reads(
        &mut self,
        lane: ScanTaskLane,
        reads: &[ScanTaskRead],
    ) -> Vec<ReadRequestKey> {
        let released_bytes = scan_task_read_bytes(reads);
        self.active_group_read_bytes[lane.group().idx()] =
            self.active_group_read_bytes[lane.group().idx()].saturating_sub(released_bytes);
        let mut released_keys = Vec::new();
        let mut seen = HashSet::new();
        for read in reads {
            if !seen.insert(read.key) {
                continue;
            }
            let Entry::Occupied(mut entry) = self.active_reads.entry(read.key) else {
                continue;
            };
            if entry.get().refs > 1 {
                entry.get_mut().refs -= 1;
            } else {
                let active = entry.remove();
                self.active_read_bytes = self.active_read_bytes.saturating_sub(active.bytes);
                released_keys.push(read.key);
            }
        }
        tracing::trace!(
            target: "vortex_scan::task",
            ?lane,
            read_count = reads.len(),
            released_bytes,
            active_read_bytes = self.active_read_bytes,
            "released scan task reads"
        );
        released_keys
    }
}

fn drop_dead_heads_from_map<K: Copy + Ord, T>(
    queues: &mut BTreeMap<K, VecDeque<ScanTaskBox<T>>>,
    should_drop: &mut impl FnMut(&ScanTaskBox<T>) -> bool,
) {
    let keys = queues.keys().copied().collect::<Vec<_>>();
    for key in keys {
        let Some(queue) = queues.get_mut(&key) else {
            continue;
        };
        while queue.front().is_some_and(&mut *should_drop) {
            queue.pop_front();
        }
        if queue.is_empty() {
            queues.remove(&key);
        }
    }
}

fn can_admit_task(
    active_read_bytes: u64,
    read_byte_budget: u64,
    in_flight_empty: bool,
    incremental_read_bytes: u64,
) -> bool {
    incremental_read_bytes == 0
        || active_read_bytes.saturating_add(incremental_read_bytes) <= read_byte_budget
        || in_flight_empty
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct TaskScore {
    priority: u64,
    incremental_read_bytes: u64,
    read_bytes: u64,
    morsel_id: usize,
    lane_idx: u32,
}

impl TaskScore {
    fn new(
        active_reads: &HashMap<ReadRequestKey, ActiveRead>,
        reads: &[ScanTaskRead],
        priority: u64,
        morsel_id: usize,
        lane_idx: u32,
    ) -> Self {
        Self {
            priority,
            incremental_read_bytes: incremental_read_bytes(active_reads, reads),
            read_bytes: scan_task_read_bytes(reads),
            morsel_id,
            lane_idx,
        }
    }
}

fn incremental_read_bytes(
    active_reads: &HashMap<ReadRequestKey, ActiveRead>,
    reads: &[ScanTaskRead],
) -> u64 {
    let mut seen = HashSet::new();
    reads
        .iter()
        .filter(|read| seen.insert(read.key) && !active_reads.contains_key(&read.key))
        .map(|read| read.bytes)
        .sum()
}

/// Count each unique task read once.
pub fn scan_task_read_bytes(reads: &[ScanTaskRead]) -> u64 {
    let mut seen = HashSet::new();
    reads
        .iter()
        .filter(|read| seen.insert(read.key))
        .map(|read| read.bytes)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    fn read(key: u32, bytes: u64) -> ScanTaskRead {
        ScanTaskRead {
            key: ReadRequestKey::new(u64::from(key)),
            bytes,
        }
    }

    fn task(morsel_id: usize, phase: ScanIoPhase, reads: Vec<ScanTaskRead>) -> ScanTaskBox<()> {
        ScanStep::ready(
            morsel_id,
            phase,
            ScanTaskLane::from_phase(phase),
            reads,
            Ok(()),
        )
        .boxed()
    }

    fn task_in_lane(
        morsel_id: usize,
        phase: ScanIoPhase,
        lane: ScanTaskLane,
        reads: Vec<ScanTaskRead>,
    ) -> ScanTaskBox<()> {
        ScanStep::ready(morsel_id, phase, lane, reads, Ok(())).boxed()
    }

    fn prioritized_task_in_lane(
        morsel_id: usize,
        phase: ScanIoPhase,
        lane: ScanTaskLane,
        reads: Vec<ScanTaskRead>,
        priority: u64,
    ) -> ScanTaskBox<()> {
        ScanStep::ready(morsel_id, phase, lane, reads, Ok(()))
            .with_priority(priority)
            .boxed()
    }

    #[test]
    fn queue_admits_by_incremental_read_budget() {
        let mut queue = ScanTaskQueue::new(100);
        queue.push(task(0, ScanIoPhase::EvidenceProbe, vec![read(1, 80)]));
        queue.push(task(0, ScanIoPhase::ProjectionRead, vec![read(2, 80)]));

        let evidence = queue
            .pop_next_admissible(true, |_| true)
            .expect("evidence task should be admitted");
        assert_eq!(queue.active_read_bytes(), 80);
        assert!(queue.pop_next_admissible(false, |_| true).is_none());

        queue.release_reads(evidence.lane(), evidence.admitted_reads());
        let projection = queue
            .pop_next_admissible(false, |_| true)
            .expect("projection task should be admitted after release");
        assert_eq!(queue.active_read_bytes(), 80);
        queue.release_reads(projection.lane(), projection.admitted_reads());
        assert_eq!(queue.active_read_bytes(), 0);
    }

    #[test]
    fn queue_dedupes_reads_within_one_task() {
        let duplicate = read(1, 40);
        let mut queue = ScanTaskQueue::new(100);
        queue.push(task(
            0,
            ScanIoPhase::ProjectionRead,
            vec![duplicate, duplicate],
        ));

        let admitted = queue
            .pop_next_admissible(true, |_| true)
            .expect("task should be admitted");
        assert_eq!(admitted.admitted_reads().len(), 1);
        assert_eq!(queue.active_read_bytes(), 40);
    }

    #[test]
    fn queue_releases_shared_read_after_last_active_ref() {
        let shared = read(1, 40);
        let mut queue = ScanTaskQueue::new(100);
        queue.push(task_in_lane(
            0,
            ScanIoPhase::PredicateRead,
            ScanTaskLane::Predicate { predicate_idx: 0 },
            vec![shared],
        ));
        queue.push(task_in_lane(
            1,
            ScanIoPhase::PredicateRead,
            ScanTaskLane::Predicate { predicate_idx: 1 },
            vec![shared],
        ));

        let first = queue
            .pop_next_admissible(true, |_| true)
            .expect("first task should be admitted");
        let second = queue
            .pop_next_admissible(false, |_| true)
            .expect("second task should reuse the active read");
        assert_eq!(queue.active_read_bytes(), 40);

        assert!(
            queue
                .release_reads(first.lane(), first.admitted_reads())
                .is_empty()
        );
        assert_eq!(queue.active_read_bytes(), 40);
        assert_eq!(
            queue.release_reads(second.lane(), second.admitted_reads()),
            vec![shared.key]
        );
        assert_eq!(queue.active_read_bytes(), 0);
    }

    #[test]
    fn queue_prefers_smaller_incremental_bytes_without_morsel_frontier() {
        let mut queue = ScanTaskQueue::new(100);
        queue.push(task_in_lane(
            0,
            ScanIoPhase::EvidenceProbe,
            ScanTaskLane::Evidence {
                predicate_idx: 0,
                evidence_idx: 0,
            },
            vec![read(1, 90)],
        ));
        queue.push(task_in_lane(
            1,
            ScanIoPhase::EvidenceProbe,
            ScanTaskLane::Evidence {
                predicate_idx: 0,
                evidence_idx: 1,
            },
            vec![read(2, 10)],
        ));

        let next = queue
            .pop_next_admissible(true, |_| true)
            .expect("one task should be admitted");
        let (task, _lane, reads) = next.into_parts();
        assert_eq!(task.morsel_id(), 1);
        assert_eq!(reads, vec![read(2, 10)]);
    }

    #[test]
    fn queue_runs_evidence_before_ready_predicate() {
        let mut queue = ScanTaskQueue::new(100);
        queue.push(task_in_lane(
            0,
            ScanIoPhase::EvidenceProbe,
            ScanTaskLane::Evidence {
                predicate_idx: 0,
                evidence_idx: 0,
            },
            vec![read(1, 10)],
        ));
        queue.push(task_in_lane(
            0,
            ScanIoPhase::PredicateRead,
            ScanTaskLane::Predicate { predicate_idx: 0 },
            vec![read(2, 10)],
        ));

        let next = queue
            .pop_next_admissible(true, |_| true)
            .expect("one task should be admitted");
        let (task, lane, reads) = next.into_parts();
        assert_eq!(task.morsel_id(), 0);
        assert_eq!(
            lane,
            ScanTaskLane::Evidence {
                predicate_idx: 0,
                evidence_idx: 0,
            }
        );
        assert_eq!(reads, vec![read(1, 10)]);
    }

    #[test]
    fn queue_projection_gate_still_allows_predicates() {
        let mut queue = ScanTaskQueue::new(100);
        queue.push(task_in_lane(
            0,
            ScanIoPhase::ProjectionRead,
            ScanTaskLane::Projection,
            vec![read(1, 10)],
        ));
        queue.push(task_in_lane(
            0,
            ScanIoPhase::PredicateRead,
            ScanTaskLane::Predicate { predicate_idx: 0 },
            vec![read(2, 10)],
        ));

        let next = queue
            .pop_next_admissible_with_projection_gate(true, false, |_| true)
            .expect("predicate task should still be admitted while projection is gated");
        let (_task, lane, reads) = next.into_parts();
        assert_eq!(lane, ScanTaskLane::Predicate { predicate_idx: 0 });
        assert_eq!(reads, vec![read(2, 10)]);

        assert!(
            queue
                .pop_next_admissible_with_projection_gate(false, false, |_| true)
                .is_none(),
            "projection task should remain queued while projection is gated"
        );
    }

    #[test]
    fn queue_keeps_scan_evidence_for_dead_anchor_morsel() {
        let mut queue = ScanTaskQueue::new(100);
        queue.push(task_in_lane(
            0,
            ScanIoPhase::EvidenceProbe,
            ScanTaskLane::ScanEvidence {
                predicate_idx: 0,
                evidence_idx: 0,
            },
            vec![read(1, 10)],
        ));

        let next = queue
            .pop_next_admissible(true, |_| false)
            .expect("scan-scope evidence task should not be dropped with its anchor morsel");
        let (task, lane, reads) = next.into_parts();
        assert_eq!(task.morsel_id(), 0);
        assert_eq!(
            lane,
            ScanTaskLane::ScanEvidence {
                predicate_idx: 0,
                evidence_idx: 0,
            }
        );
        assert_eq!(reads, vec![read(1, 10)]);
    }

    #[test]
    fn queue_prefers_lower_priority_within_group() {
        let mut queue = ScanTaskQueue::new(100);
        queue.push(prioritized_task_in_lane(
            0,
            ScanIoPhase::PredicateRead,
            ScanTaskLane::Predicate { predicate_idx: 0 },
            vec![read(1, 10)],
            100,
        ));
        queue.push(prioritized_task_in_lane(
            0,
            ScanIoPhase::PredicateRead,
            ScanTaskLane::Predicate { predicate_idx: 1 },
            vec![read(2, 10)],
            10,
        ));

        let next = queue
            .pop_next_admissible(true, |_| true)
            .expect("one task should be admitted");
        let (task, lane, reads) = next.into_parts();
        assert_eq!(task.morsel_id(), 0);
        assert_eq!(lane, ScanTaskLane::Predicate { predicate_idx: 1 });
        assert_eq!(reads, vec![read(2, 10)]);
    }
}
