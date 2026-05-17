// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Partition-local scheduler queue, priorities, and task types.

use std::collections::BinaryHeap;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;
use std::task::Context;
use std::task::Poll;

use futures::FutureExt;
use futures::future::poll_fn;
use vortex_error::VortexError;
use vortex_error::VortexResult;

use crate::segments::SegmentId;
use crate::v2::domain::DomainId;
use crate::v2::experiment::trace_flow;
use crate::v2::plans::flat::SharedSegmentFuture;

/// Stable identifier for one DataFusion partition-local scheduler.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct PartitionSchedulerId(u32);

impl PartitionSchedulerId {
    /// Construct a partition scheduler identifier.
    pub(crate) const fn new(id: u32) -> Self {
        Self(id)
    }

    pub(super) const fn raw(self) -> u32 {
        self.0
    }
}

/// Stable identifier for a pipeline inside one partition scheduler.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct PipelineId(usize);

impl PipelineId {
    /// Construct a pipeline identifier.
    pub(crate) const fn new(id: usize) -> Self {
        Self(id)
    }

    /// Index into the owning scheduler's pipeline state table.
    pub(crate) const fn index(self) -> usize {
        self.0
    }
}

/// Stable identifier for a morsel.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct MorselId(u64);

impl MorselId {
    /// Construct a morsel identifier.
    pub(crate) const fn new(id: u64) -> Self {
        Self(id)
    }

    pub(super) const fn raw(self) -> u64 {
        self.0
    }
}

/// Stable identifier for an I/O request tracked by the scheduler.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Ord, PartialOrd)]
pub(crate) struct IoRequestId(u64);

impl IoRequestId {
    /// Construct an I/O request identifier.
    pub(crate) const fn new(id: u64) -> Self {
        Self(id)
    }

    pub(super) const fn raw(self) -> u64 {
        self.0
    }
}

/// Role a morsel plays for scheduling.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub(crate) enum MorselRole {
    /// Produces runtime information such as row demand or SIP filters.
    InformationProducer,
    /// Consumes information to decide whether value work is needed.
    InformationConsumer,
    /// Produces projected values.
    ValueProducer,
    /// Combines sibling morsels or masks.
    Combiner,
    /// Sits on an output retirement path.
    Sink,
}

impl MorselRole {
    pub(super) const fn as_str(self) -> &'static str {
        match self {
            Self::InformationProducer => "information_producer",
            Self::InformationConsumer => "information_consumer",
            Self::ValueProducer => "value_producer",
            Self::Combiner => "combiner",
            Self::Sink => "sink",
        }
    }
}

/// Source operator carried by a lowered scheduler pipeline.
pub(crate) trait SchedulerSourceNode: Send + Sync + 'static {
    /// Human-readable operator label for traces and diagnostics.
    fn label(&self) -> &str;

    /// Scheduling role for the first morsel admitted for this source.
    fn role(&self) -> MorselRole;
}

/// Transform operator carried by a lowered scheduler pipeline.
pub(crate) trait SchedulerTransformNode: Send + Sync + 'static {
    /// Human-readable operator label for traces and diagnostics.
    fn label(&self) -> &str;
}

/// Sink operator that terminates a lowered scheduler pipeline.
pub(crate) trait SchedulerSinkNode: Send + Sync + 'static {
    /// Human-readable operator label for traces and diagnostics.
    fn label(&self) -> &str;
}

/// Source that closes a lowered pipeline.
#[derive(Clone)]
pub(crate) struct SchedulerPipelineSource {
    node: Arc<dyn SchedulerSourceNode>,
}

impl SchedulerPipelineSource {
    pub(crate) fn new(node: impl SchedulerSourceNode) -> Self {
        Self {
            node: Arc::new(node),
        }
    }

    pub(crate) fn label(&self) -> &str {
        self.node.label()
    }

    pub(crate) fn role(&self) -> MorselRole {
        self.node.role()
    }
}

impl fmt::Debug for SchedulerPipelineSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SchedulerPipelineSource")
            .field("label", &self.label())
            .field("role", &self.role())
            .finish()
    }
}

/// Transform carried by a lowered pipeline.
#[derive(Clone)]
pub(crate) struct SchedulerPipelineTransform {
    node: Arc<dyn SchedulerTransformNode>,
}

impl SchedulerPipelineTransform {
    pub(crate) fn new(node: impl SchedulerTransformNode) -> Self {
        Self {
            node: Arc::new(node),
        }
    }

    pub(crate) fn label(&self) -> &str {
        self.node.label()
    }
}

impl fmt::Debug for SchedulerPipelineTransform {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SchedulerPipelineTransform")
            .field("label", &self.label())
            .finish()
    }
}

/// Sink that terminates a lowered pipeline.
#[derive(Clone)]
pub(crate) struct SchedulerPipelineSink {
    node: Arc<dyn SchedulerSinkNode>,
}

impl SchedulerPipelineSink {
    pub(crate) fn new(node: impl SchedulerSinkNode) -> Self {
        Self {
            node: Arc::new(node),
        }
    }

    pub(crate) fn label(&self) -> &str {
        self.node.label()
    }
}

impl fmt::Debug for SchedulerPipelineSink {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SchedulerPipelineSink")
            .field("label", &self.label())
            .finish()
    }
}

/// Scheduler-local state for one lowered pipeline.
#[derive(Clone, Debug)]
pub(crate) struct SchedulerPipelineState {
    source: SchedulerPipelineSource,
    transforms: Vec<SchedulerPipelineTransform>,
    sink: SchedulerPipelineSink,
}

impl SchedulerPipelineState {
    fn new(
        source: SchedulerPipelineSource,
        transforms: Vec<SchedulerPipelineTransform>,
        sink: SchedulerPipelineSink,
    ) -> Self {
        Self {
            source,
            transforms,
            sink,
        }
    }

    pub(super) fn source(&self) -> &SchedulerPipelineSource {
        &self.source
    }

    pub(super) fn transforms(&self) -> &[SchedulerPipelineTransform] {
        &self.transforms
    }

    pub(super) fn sink(&self) -> &SchedulerPipelineSink {
        &self.sink
    }
}

/// Cost and memory estimates for one morsel.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MorselEstimate {
    pub(super) cpu_ns: u64,
    pub(super) io_bytes: u64,
    memory_bytes: u64,
}

impl MorselEstimate {
    /// Construct a morsel estimate.
    pub(crate) const fn new(cpu_ns: u64, io_bytes: u64, memory_bytes: u64) -> Self {
        Self {
            cpu_ns,
            io_bytes,
            memory_bytes,
        }
    }

    /// Estimated queued memory.
    pub(crate) fn memory_bytes(self) -> u64 {
        self.memory_bytes
    }
}

/// Integer version of the information-priority score.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct MorselPriority {
    readiness: i64,
    info_gain_ns: u64,
    cost_ns: u64,
    lead_credit: i64,
    age_credit_per_tick: i64,
    memory_pressure: i64,
    backpressure: i64,
}

impl MorselPriority {
    /// Construct a priority from the score terms.
    #[allow(clippy::too_many_arguments)]
    pub(crate) const fn new(
        readiness: i64,
        info_gain_ns: u64,
        cost_ns: u64,
        lead_credit: i64,
        age_credit_per_tick: i64,
        memory_pressure: i64,
        backpressure: i64,
    ) -> Self {
        Self {
            readiness,
            info_gain_ns,
            cost_ns,
            lead_credit,
            age_credit_per_tick,
            memory_pressure,
            backpressure,
        }
    }

    /// Priority for a demand/SIP producer that should run ahead.
    pub(crate) fn information_producer(info_gain_ns: u64, cost_ns: u64, lead_credit: i64) -> Self {
        Self::new(0, info_gain_ns, cost_ns, lead_credit, 1, 0, 0)
    }

    /// Priority for ordinary value work.
    pub(crate) fn value_work(info_gain_ns: u64, cost_ns: u64, backpressure: i64) -> Self {
        Self::new(0, info_gain_ns, cost_ns, 0, 1, 0, backpressure)
    }

    /// Score at `now_tick`, scaled by 1024 to avoid floating point
    /// ordering inside the heap.
    fn score(self, ready_tick: u64, now_tick: u64) -> i64 {
        const SCALE: u128 = 1024;
        let ratio = if self.cost_ns == 0 {
            i64::MAX / 4
        } else {
            let scaled =
                (u128::from(self.info_gain_ns).saturating_mul(SCALE)) / u128::from(self.cost_ns);
            let capped = scaled.min((i64::MAX / 4) as u128);
            i64::try_from(capped).unwrap_or(i64::MAX / 4)
        };
        let age = now_tick.saturating_sub(ready_tick).min(i64::MAX as u64) as i64;
        self.readiness
            .saturating_add(ratio)
            .saturating_add(self.lead_credit)
            .saturating_add(age.saturating_mul(self.age_credit_per_tick))
            .saturating_sub(self.memory_pressure)
            .saturating_sub(self.backpressure)
    }
}

/// One schedulable data morsel.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SchedulerMorsel {
    pub(super) id: MorselId,
    pub(super) domain: DomainId,
    pub(super) order_key: Range<u64>,
    pub(super) role: MorselRole,
    pub(super) stage: u16,
    pub(super) stage_count: u16,
    pub(super) estimate: MorselEstimate,
    pub(super) priority: MorselPriority,
}

impl SchedulerMorsel {
    /// Construct a scheduler morsel.
    #[allow(clippy::too_many_arguments)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        id: MorselId,
        domain: DomainId,
        order_key: Range<u64>,
        role: MorselRole,
        stage_count: u16,
        estimate: MorselEstimate,
        priority: MorselPriority,
    ) -> Self {
        Self {
            id,
            domain,
            order_key,
            role,
            stage: 0,
            stage_count: stage_count.max(1),
            estimate,
            priority,
        }
    }

    /// Morsel id.
    pub(crate) fn id(&self) -> MorselId {
        self.id
    }

    /// Current pipeline stage.
    pub(crate) fn stage(&self) -> u16 {
        self.stage
    }

    pub(super) fn advance_one_stage(&mut self) -> Option<(u16, u16)> {
        let from = self.stage;
        if self.stage + 1 >= self.stage_count {
            return None;
        }
        self.stage += 1;
        Some((from, self.stage))
    }

    fn memory_bytes(&self) -> u64 {
        self.estimate.memory_bytes()
    }
}

/// CPU/operator work tracked by the scheduler.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct SchedulerWorkTask {
    pub(super) pipeline: PipelineId,
    pub(super) morsel: SchedulerMorsel,
}

impl SchedulerWorkTask {
    /// Construct a work task.
    pub(crate) fn new(pipeline: PipelineId, morsel: SchedulerMorsel) -> Self {
        Self { pipeline, morsel }
    }

    fn memory_bytes(&self) -> u64 {
        self.morsel.memory_bytes()
    }
}

/// Segment future tracked in the same priority queue as CPU morsels.
pub(crate) struct SchedulerSegmentTask {
    pub(super) id: IoRequestId,
    pub(super) pipeline: PipelineId,
    pub(super) segment_id: SegmentId,
    pub(super) domain: DomainId,
    pub(super) range: Range<u64>,
    pub(super) bytes: u64,
    pub(super) priority: MorselPriority,
    segment_future: Option<SharedSegmentFuture>,
}

impl fmt::Debug for SchedulerSegmentTask {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("SchedulerSegmentTask")
            .field("id", &self.id)
            .field("pipeline", &self.pipeline)
            .field("segment_id", &self.segment_id)
            .field("domain", &self.domain)
            .field("range", &self.range)
            .field("bytes", &self.bytes)
            .field("has_segment_future", &self.segment_future.is_some())
            .finish_non_exhaustive()
    }
}

impl SchedulerSegmentTask {
    /// Construct a segment task.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        id: IoRequestId,
        pipeline: PipelineId,
        segment_id: SegmentId,
        domain: DomainId,
        range: Range<u64>,
        bytes: u64,
        priority: MorselPriority,
        segment_future: SharedSegmentFuture,
    ) -> Self {
        Self {
            id,
            pipeline,
            segment_id,
            domain,
            range,
            bytes,
            priority,
            segment_future: Some(segment_future),
        }
    }

    #[cfg(test)]
    pub(super) fn metadata_only(
        id: IoRequestId,
        pipeline: PipelineId,
        segment_id: SegmentId,
        domain: DomainId,
        range: Range<u64>,
        bytes: u64,
        priority: MorselPriority,
    ) -> Self {
        Self {
            id,
            pipeline,
            segment_id,
            domain,
            range,
            bytes,
            priority,
            segment_future: None,
        }
    }

    fn poll(&mut self, cx: &mut Context<'_>) -> Poll<VortexResult<u64>> {
        let Some(segment_future) = &mut self.segment_future else {
            return Poll::Ready(Ok(self.bytes));
        };
        match segment_future.poll_unpin(cx) {
            Poll::Pending => Poll::Pending,
            Poll::Ready(result) => {
                drop(result.map_err(VortexError::from)?);
                self.segment_future = None;
                Poll::Ready(Ok(self.bytes))
            }
        }
    }

    pub(super) async fn wait(&mut self) -> VortexResult<u64> {
        poll_fn(|cx| self.poll(cx)).await
    }
}

/// Work-stealing and balancing control tasks.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SchedulerControlEvent {
    /// Ask another partition scheduler for work.
    StealRequest {
        from: PartitionSchedulerId,
        target_bytes: u64,
    },
    /// Offer work to another partition scheduler.
    StealOffer {
        from: PartitionSchedulerId,
        offered_morsels: u32,
    },
    /// Recompute priorities and budgets.
    Rebalance { reason: &'static str },
}

/// Task stored in a partition-local scheduler queue.
///
/// This is deliberately an enum rather than a boxed trait object: the
/// scheduler can switch on a small set of runtime work units without a
/// per-task vtable allocation.
#[derive(Debug)]
pub(crate) enum SchedulerTask {
    /// CPU or operator work.
    Work(SchedulerWorkTask),
    /// Segment I/O future.
    Segment(SchedulerSegmentTask),
    /// Work-stealing or balancing task.
    Control(SchedulerControlEvent),
}

impl SchedulerTask {
    fn memory_bytes(&self) -> u64 {
        match self {
            SchedulerTask::Work(work) => work.memory_bytes(),
            SchedulerTask::Segment(_) | SchedulerTask::Control(_) => 0,
        }
    }

    fn priority(&self, ready_tick: u64, now_tick: u64) -> QueuePriority {
        match self {
            SchedulerTask::Work(work) => QueuePriority::new(
                event_class_for_role(work.morsel.role),
                work.morsel.priority.score(ready_tick, now_tick),
            ),
            SchedulerTask::Segment(request) => {
                QueuePriority::new(EventClass::Io, request.priority.score(ready_tick, now_tick))
            }
            SchedulerTask::Control(control) => QueuePriority::new(
                EventClass::Control,
                match control {
                    SchedulerControlEvent::Rebalance { .. } => 1_000,
                    SchedulerControlEvent::StealRequest { .. } => 500,
                    SchedulerControlEvent::StealOffer { .. } => 250,
                },
            ),
        }
    }
}

#[derive(Clone, Copy)]
struct TaskTraceFields {
    kind: &'static str,
    pipeline: i64,
    morsel_id: u64,
    request_id: u64,
    role: &'static str,
    row_start: u64,
    row_end: u64,
    stage: u16,
    stage_count: u16,
    memory_bytes: u64,
}

impl TaskTraceFields {
    const fn empty(kind: &'static str) -> Self {
        Self {
            kind,
            pipeline: -1,
            morsel_id: 0,
            request_id: 0,
            role: "none",
            row_start: 0,
            row_end: 0,
            stage: 0,
            stage_count: 0,
            memory_bytes: 0,
        }
    }
}

fn task_trace_fields(task: &SchedulerTask) -> TaskTraceFields {
    match task {
        SchedulerTask::Work(work) => TaskTraceFields {
            kind: "work",
            pipeline: work.pipeline.index() as i64,
            morsel_id: work.morsel.id.raw(),
            request_id: 0,
            role: work.morsel.role.as_str(),
            row_start: work.morsel.order_key.start,
            row_end: work.morsel.order_key.end,
            stage: work.morsel.stage,
            stage_count: work.morsel.stage_count,
            memory_bytes: work.memory_bytes(),
        },
        SchedulerTask::Segment(segment) => TaskTraceFields {
            kind: "segment",
            pipeline: segment.pipeline.index() as i64,
            morsel_id: 0,
            request_id: segment.id.raw(),
            role: "io",
            row_start: segment.range.start,
            row_end: segment.range.end,
            stage: 0,
            stage_count: 0,
            memory_bytes: 0,
        },
        SchedulerTask::Control(_) => TaskTraceFields::empty("control"),
    }
}

fn event_class_for_role(role: MorselRole) -> EventClass {
    match role {
        MorselRole::Sink => EventClass::RetirementCritical,
        MorselRole::InformationProducer => EventClass::Information,
        MorselRole::InformationConsumer | MorselRole::ValueProducer | MorselRole::Combiner => {
            EventClass::Data
        }
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd)]
enum EventClass {
    Control = 0,
    Io = 1,
    Data = 2,
    Information = 3,
    RetirementCritical = 4,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct QueuePriority {
    class: EventClass,
    score: i64,
}

impl QueuePriority {
    fn new(class: EventClass, score: i64) -> Self {
        Self { class, score }
    }
}

impl Ord for QueuePriority {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.class
            .cmp(&other.class)
            .then_with(|| self.score.cmp(&other.score))
    }
}

impl PartialOrd for QueuePriority {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug)]
struct QueueEntry {
    priority: QueuePriority,
    sequence: u64,
    ready_tick: u64,
    task: SchedulerTask,
}

impl PartialEq for QueueEntry {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
            && self.sequence == other.sequence
            && self.ready_tick == other.ready_tick
    }
}

impl Eq for QueueEntry {}

impl Ord for QueueEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority
            .cmp(&other.priority)
            // BinaryHeap is max-first; reverse sequence for FIFO
            // behavior among equally-ranked events.
            .then_with(|| other.sequence.cmp(&self.sequence))
    }
}

impl PartialOrd for QueueEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// Queue bounds for one partition scheduler.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct SchedulerBudget {
    max_queued_events: usize,
    max_queued_memory_bytes: u64,
}

impl SchedulerBudget {
    /// Construct scheduler queue bounds.
    pub(crate) const fn new(max_queued_events: usize, max_queued_memory_bytes: u64) -> Self {
        Self {
            max_queued_events,
            max_queued_memory_bytes,
        }
    }
}

impl Default for SchedulerBudget {
    fn default() -> Self {
        Self::new(1024, 64 * 1024 * 1024)
    }
}

/// Result of letting one partition scheduler make progress once.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) enum SchedulerStep {
    /// A morsel advanced one stage and was requeued if needed.
    Advanced {
        morsel_id: MorselId,
        pipeline: PipelineId,
        from_stage: u16,
        to_stage: u16,
    },
    /// A morsel completed its last stage.
    Completed {
        morsel_id: MorselId,
        pipeline: PipelineId,
    },
    /// The scheduler completed a segment future.
    CompletedSegment {
        request_id: IoRequestId,
        pipeline: PipelineId,
        segment_id: SegmentId,
        bytes: u64,
    },
    /// The scheduler polled a segment future that was not ready.
    PendingSegment {
        request_id: IoRequestId,
        pipeline: PipelineId,
        segment_id: SegmentId,
    },
    /// The scheduler handled a balancing/work-stealing event.
    Control { event: SchedulerControlEvent },
}

/// Prototype partition-local scheduler.
///
/// The intended runtime shape is one scheduler per DataFusion output
/// partition. Each scheduler owns a priority queue containing data
/// morsels, I/O requests, and balancing events. Calling
/// [`Self::make_progress`] advances exactly one event.
#[derive(Debug)]
pub(crate) struct PartitionScheduler {
    id: PartitionSchedulerId,
    budget: SchedulerBudget,
    pipelines: Vec<SchedulerPipelineState>,
    queue: BinaryHeap<QueueEntry>,
    next_sequence: u64,
    queued_memory_bytes: u64,
}

impl PartitionScheduler {
    /// Construct a partition scheduler.
    pub(crate) fn new(id: PartitionSchedulerId, budget: SchedulerBudget) -> Self {
        Self {
            id,
            budget,
            pipelines: Vec::new(),
            queue: BinaryHeap::new(),
            next_sequence: 0,
            queued_memory_bytes: 0,
        }
    }

    /// Scheduler id.
    pub(crate) fn id(&self) -> PartitionSchedulerId {
        self.id
    }

    /// Number of queued events.
    pub(crate) fn len(&self) -> usize {
        self.queue.len()
    }

    /// Queued morsel memory.
    pub(crate) fn queued_memory_bytes(&self) -> u64 {
        self.queued_memory_bytes
    }

    /// Number of closed pipelines owned by this scheduler.
    pub(crate) fn pipeline_count(&self) -> usize {
        self.pipelines.len()
    }

    /// Close a lowered pipeline by attaching its source and return
    /// the scheduler-local opaque id.
    pub(crate) fn close_pipeline_with_source(
        &mut self,
        source: SchedulerPipelineSource,
        transforms: Vec<SchedulerPipelineTransform>,
        sink: SchedulerPipelineSink,
    ) -> PipelineId {
        let id = PipelineId::new(self.pipelines.len());
        self.pipelines
            .push(SchedulerPipelineState::new(source, transforms, sink));
        id
    }

    pub(super) fn pipeline_source(&self, pipeline: PipelineId) -> Option<&SchedulerPipelineSource> {
        self.pipelines
            .get(pipeline.index())
            .map(SchedulerPipelineState::source)
    }

    pub(super) fn pipeline_state(&self, pipeline: PipelineId) -> Option<&SchedulerPipelineState> {
        self.pipelines.get(pipeline.index())
    }

    pub(super) fn pop_task(&mut self, now_tick: u64) -> Option<SchedulerTask> {
        self.refresh_priorities(now_tick);
        let entry = self.queue.pop()?;
        if trace_flow() {
            let fields = task_trace_fields(&entry.task);
            tracing::debug!(
                target: "vortex_layout::v2::flow",
                scheduler_id = self.id.raw(),
                tick = now_tick,
                task_kind = fields.kind,
                pipeline = fields.pipeline,
                morsel_id = fields.morsel_id,
                request_id = fields.request_id,
                role = fields.role,
                row_start = fields.row_start,
                row_end = fields.row_end,
                rows = fields.row_end.saturating_sub(fields.row_start),
                stage = fields.stage,
                stage_count = fields.stage_count,
                queue_len_after = self.queue.len(),
                queued_memory_bytes = self
                    .queued_memory_bytes
                    .saturating_sub(entry.task.memory_bytes()),
                priority_class = ?entry.priority.class,
                priority_score = entry.priority.score,
                "scheduler pop"
            );
        }
        self.queued_memory_bytes = self
            .queued_memory_bytes
            .saturating_sub(entry.task.memory_bytes());
        Some(entry.task)
    }

    /// Enqueue a task. Returns `false` when the bounded queue would
    /// exceed its task or memory budget.
    pub(crate) fn enqueue(&mut self, task: SchedulerTask, now_tick: u64) -> bool {
        let memory_bytes = task.memory_bytes();
        if self.queue.len() >= self.budget.max_queued_events
            || self.queued_memory_bytes.saturating_add(memory_bytes)
                > self.budget.max_queued_memory_bytes
        {
            return false;
        }

        let sequence = self.next_sequence;
        self.next_sequence = self.next_sequence.saturating_add(1);
        let priority = task.priority(now_tick, now_tick);
        if trace_flow() {
            let fields = task_trace_fields(&task);
            tracing::debug!(
                target: "vortex_layout::v2::flow",
                scheduler_id = self.id.raw(),
                tick = now_tick,
                task_kind = fields.kind,
                pipeline = fields.pipeline,
                morsel_id = fields.morsel_id,
                request_id = fields.request_id,
                role = fields.role,
                row_start = fields.row_start,
                row_end = fields.row_end,
                rows = fields.row_end.saturating_sub(fields.row_start),
                stage = fields.stage,
                stage_count = fields.stage_count,
                queue_len_before = self.queue.len(),
                queue_len_after = self.queue.len() + 1,
                queued_memory_bytes = self.queued_memory_bytes.saturating_add(memory_bytes),
                priority_class = ?priority.class,
                priority_score = priority.score,
                "scheduler enqueue"
            );
        }
        self.queued_memory_bytes = self.queued_memory_bytes.saturating_add(memory_bytes);
        self.queue.push(QueueEntry {
            priority,
            sequence,
            ready_tick: now_tick,
            task,
        });
        true
    }

    /// Recompute priority keys, mainly to age long-waiting morsels and
    /// incorporate changed scheduler state after balancing.
    pub(crate) fn refresh_priorities(&mut self, now_tick: u64) {
        let entries = std::mem::take(&mut self.queue);
        self.queue = entries
            .into_iter()
            .map(|mut entry| {
                entry.priority = entry.task.priority(entry.ready_tick, now_tick);
                entry
            })
            .collect();
    }

    /// Advance one task.
    pub(crate) fn make_progress(&mut self, now_tick: u64) -> Option<SchedulerStep> {
        let waker = futures::task::noop_waker();
        let mut cx = Context::from_waker(&waker);
        self.poll_progress(now_tick, &mut cx)
    }

    /// Poll one scheduler task.
    pub(crate) fn poll_progress(
        &mut self,
        now_tick: u64,
        cx: &mut Context<'_>,
    ) -> Option<SchedulerStep> {
        match self.pop_task(now_tick)? {
            SchedulerTask::Work(mut work) => {
                let morsel_id = work.morsel.id;
                let pipeline = work.pipeline;
                if let Some((from_stage, to_stage)) = work.morsel.advance_one_stage() {
                    let _requeued = self.enqueue(SchedulerTask::Work(work), now_tick);
                    Some(SchedulerStep::Advanced {
                        morsel_id,
                        pipeline,
                        from_stage,
                        to_stage,
                    })
                } else {
                    Some(SchedulerStep::Completed {
                        morsel_id,
                        pipeline,
                    })
                }
            }
            SchedulerTask::Segment(mut request) => match request.poll(cx) {
                Poll::Ready(Ok(bytes)) => Some(SchedulerStep::CompletedSegment {
                    request_id: request.id,
                    pipeline: request.pipeline,
                    segment_id: request.segment_id,
                    bytes,
                }),
                Poll::Ready(Err(_err)) => Some(SchedulerStep::CompletedSegment {
                    request_id: request.id,
                    pipeline: request.pipeline,
                    segment_id: request.segment_id,
                    bytes: request.bytes,
                }),
                Poll::Pending => {
                    let request_id = request.id;
                    let pipeline = request.pipeline;
                    let segment_id = request.segment_id;
                    let _requeued = self.enqueue(SchedulerTask::Segment(request), now_tick);
                    Some(SchedulerStep::PendingSegment {
                        request_id,
                        pipeline,
                        segment_id,
                    })
                }
            },
            SchedulerTask::Control(event) => Some(SchedulerStep::Control { event }),
        }
    }

    /// Drain up to `max_morsels` lower-priority data morsels for a
    /// future work-stealing implementation.
    pub(crate) fn stealable_morsels(&mut self, max_morsels: usize) -> Vec<SchedulerMorsel> {
        let mut kept = BinaryHeap::new();
        let mut stolen = Vec::new();
        while let Some(entry) = self.queue.pop() {
            let QueueEntry {
                priority,
                sequence,
                ready_tick,
                task,
            } = entry;
            match task {
                SchedulerTask::Work(work)
                    if stolen.len() < max_morsels
                        && !matches!(
                            work.morsel.role,
                            MorselRole::Sink | MorselRole::InformationProducer
                        ) =>
                {
                    self.queued_memory_bytes =
                        self.queued_memory_bytes.saturating_sub(work.memory_bytes());
                    stolen.push(work.morsel);
                }
                task => kept.push(QueueEntry {
                    priority,
                    sequence,
                    ready_tick,
                    task,
                }),
            }
        }
        self.queue = kept;
        stolen
    }
}
