// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Layout-plan lowering for the single-scheduler prototype.

#![allow(clippy::cognitive_complexity)]

use std::ops::Range;

use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::segments::SegmentId;
use crate::v2::domain::DomainId;
use crate::v2::domain::OperatorId;
use crate::v2::plans::flat::SharedSegmentFuture;
use crate::v2::scheduler::queue::*;

/// One initial work item produced when a lowered pipeline is closed.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct InitialPipelineWork {
    source_label: String,
    operator: OperatorId,
    pipeline: PipelineId,
    morsel: MorselId,
    local_range: Range<u64>,
    global_range: Range<u64>,
    role: MorselRole,
    schema: String,
}

impl InitialPipelineWork {
    pub(crate) fn source_label(&self) -> &str {
        &self.source_label
    }

    pub(crate) fn pipeline(&self) -> PipelineId {
        self.pipeline
    }

    pub(crate) fn morsel(&self) -> MorselId {
        self.morsel
    }

    pub(crate) fn local_range(&self) -> &Range<u64> {
        &self.local_range
    }

    pub(crate) fn role(&self) -> MorselRole {
        self.role
    }

    pub(crate) fn global_range(&self) -> &Range<u64> {
        &self.global_range
    }
}

#[derive(Clone, Debug)]
struct OpenPipeline {
    transforms_rev: Vec<SchedulerPipelineTransform>,
    sink: SchedulerPipelineSink,
}

impl OpenPipeline {
    fn new(sink: SchedulerPipelineSink) -> Self {
        Self {
            transforms_rev: Vec::new(),
            sink,
        }
    }

    fn prepend_transform(&mut self, transform: SchedulerPipelineTransform) {
        self.transforms_rev.push(transform);
    }

    fn into_parts(self) -> (Vec<SchedulerPipelineTransform>, SchedulerPipelineSink) {
        let mut transforms = self.transforms_rev;
        transforms.reverse();
        (transforms, self.sink)
    }
}

struct RootSinkNode {
    label: String,
}

impl SchedulerSinkNode for RootSinkNode {
    fn label(&self) -> &str {
        &self.label
    }
}

struct ResourceSinkNode {
    label: String,
}

impl SchedulerSinkNode for ResourceSinkNode {
    fn label(&self) -> &str {
        &self.label
    }
}

struct PlanTransformNode {
    label: String,
}

impl SchedulerTransformNode for PlanTransformNode {
    fn label(&self) -> &str {
        &self.label
    }
}

struct PrototypeSourceNode {
    label: String,
    role: MorselRole,
}

impl SchedulerSourceNode for PrototypeSourceNode {
    fn label(&self) -> &str {
        &self.label
    }

    fn role(&self) -> MorselRole {
        self.role
    }
}

/// Summary returned by driving the single-scheduler prototype.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct LayoutSchedulerRunReport {
    steps: usize,
    advanced_morsels: usize,
    completed_morsels: usize,
    completed_segments: usize,
    pending_segments: usize,
    control_events: usize,
}

impl LayoutSchedulerRunReport {
    /// Number of scheduler steps executed.
    pub fn steps(&self) -> usize {
        self.steps
    }

    /// Number of morsels completed.
    pub fn completed_morsels(&self) -> usize {
        self.completed_morsels
    }

    /// Number of segment futures completed.
    pub fn completed_segments(&self) -> usize {
        self.completed_segments
    }

    /// Number of segment futures that were polled but not ready.
    pub fn pending_segments(&self) -> usize {
        self.pending_segments
    }
}

/// Lowering context for the single-scheduler layout prototype.
///
/// This maps the recursive [`crate::v2::plans::LayoutPlan`] tree into
/// the scheduler sketch above. Layout nodes lower into open pipelines:
/// transform nodes are pushed into the currently open pipeline,
/// multi-input nodes open resource pipelines, and source nodes close
/// the current pipeline into scheduler-owned pipeline state.
pub struct LayoutLoweringCtx {
    scheduler: PartitionScheduler,
    domain: DomainId,
    current_global_range: Range<u64>,
    next_operator: u32,
    next_morsel: u64,
    next_io_request: u64,
    open_pipeline: Option<OpenPipeline>,
    initial_work: Vec<InitialPipelineWork>,
}

impl LayoutLoweringCtx {
    /// Construct a lowering context for one scheduler over one ordinal
    /// row domain.
    pub fn for_single_scheduler(total_rows: u64) -> Self {
        Self::with_budget(total_rows, SchedulerBudget::default())
    }

    pub(crate) fn with_budget(total_rows: u64, budget: SchedulerBudget) -> Self {
        Self {
            scheduler: PartitionScheduler::new(PartitionSchedulerId::new(0), budget),
            domain: DomainId::new(0),
            current_global_range: 0..total_rows,
            next_operator: 1,
            next_morsel: 1,
            next_io_request: 1,
            open_pipeline: None,
            initial_work: Vec::new(),
        }
    }

    /// Run a lowering step while mapping the callee's local
    /// coordinates to `global_range` in the root scheduler domain.
    pub(crate) fn with_global_range<R>(
        &mut self,
        global_range: Range<u64>,
        f: impl FnOnce(&mut Self) -> VortexResult<R>,
    ) -> VortexResult<R> {
        let previous = std::mem::replace(&mut self.current_global_range, global_range);
        let result = f(self);
        self.current_global_range = previous;
        result
    }

    pub(crate) fn current_global_range(&self) -> Range<u64> {
        self.current_global_range.clone()
    }

    /// Open the final output pipeline for one scheduler partition.
    pub(crate) fn open_root_pipeline(&mut self, global_range: Range<u64>, schema: &DType) {
        debug_assert!(
            self.open_pipeline.is_none(),
            "attempted to open a root pipeline while another pipeline is open"
        );
        self.open_pipeline = Some(OpenPipeline::new(SchedulerPipelineSink::new(
            RootSinkNode {
                label: format!("root:{global_range:?}:{}", schema),
            },
        )));
    }

    /// Run `f` with a resource sink open for one input of `operator`.
    pub(crate) fn with_input_resource_pipeline<R>(
        &mut self,
        operator: OperatorId,
        input: usize,
        local_range: Range<u64>,
        schema: &DType,
        f: impl FnOnce(&mut Self) -> VortexResult<R>,
    ) -> VortexResult<R> {
        let previous = self.open_pipeline.take();
        let global_range = self.current_global_range.clone();
        self.open_pipeline = Some(OpenPipeline::new(SchedulerPipelineSink::new(
            ResourceSinkNode {
                label: format!(
                    "resource:operator{}:input{input}:{local_range:?}->{global_range:?}:{}",
                    operator.raw(),
                    schema
                ),
            },
        )));

        let result = f(self);
        if result.is_ok() && self.open_pipeline.is_some() {
            self.open_pipeline = previous;
            vortex_bail!("resource pipeline for input {input} of {operator:?} was not closed");
        }
        self.open_pipeline = previous;
        result
    }

    /// Prepend a plan-node transform to the currently open pipeline.
    pub(crate) fn push_plan_node(
        &mut self,
        operator: OperatorId,
        local_range: Range<u64>,
        schema: &DType,
        child_count: usize,
    ) -> VortexResult<()> {
        let Some(open_pipeline) = self.open_pipeline.as_mut() else {
            vortex_bail!("cannot push plan node without an open pipeline");
        };
        let global_range = self.current_global_range.clone();
        open_pipeline.prepend_transform(SchedulerPipelineTransform::new(PlanTransformNode {
            label: format!(
                "operator{}:{local_range:?}->{global_range:?}:{}:{child_count}children",
                operator.raw(),
                schema
            ),
        }));
        Ok(())
    }

    /// Close the current pipeline with a prototype leaf source and admit
    /// its first morsel to the scheduler.
    pub(crate) fn close_leaf_pipeline(
        &mut self,
        operator: OperatorId,
        local_range: Range<u64>,
        schema: &DType,
    ) -> VortexResult<()> {
        let role = role_for_schema(schema);
        let global_range = self.current_global_range.clone();
        let source_label = format!(
            "operator{}:leaf:{local_range:?}->{global_range:?}:{}",
            operator.raw(),
            schema
        );
        let pipeline =
            self.close_pipeline_with_source(SchedulerPipelineSource::new(PrototypeSourceNode {
                label: source_label.clone(),
                role,
            }))?;
        let morsel_id = self.alloc_morsel();
        let estimate = estimate_for_leaf(&global_range, schema);
        let priority = priority_for_leaf(role, &global_range, estimate);
        let stage_count = stage_count_for_role(role);
        let morsel = SchedulerMorsel::new(
            morsel_id,
            self.domain,
            global_range.clone(),
            role,
            stage_count,
            estimate,
            priority,
        );
        let work = SchedulerWorkTask::new(pipeline, morsel);

        if !self.scheduler.enqueue(SchedulerTask::Work(work), 0) {
            vortex_bail!(
                "layout scheduler queue full while admitting leaf pipeline work for {global_range:?}"
            );
        }

        self.initial_work.push(InitialPipelineWork {
            source_label,
            operator,
            pipeline,
            morsel: morsel_id,
            local_range,
            global_range,
            role,
            schema: schema.to_string(),
        });
        Ok(())
    }

    /// Close the current pipeline with a synthetic source for a plan node
    /// whose inputs were lowered into resource pipelines.
    pub(crate) fn close_node_output_pipeline(
        &mut self,
        operator: OperatorId,
        local_range: Range<u64>,
        schema: &DType,
        child_count: usize,
    ) -> VortexResult<MorselId> {
        let global_range = self.current_global_range.clone();
        let role = MorselRole::Combiner;
        let source_label = format!(
            "operator{}:node-output:{local_range:?}->{global_range:?}:{}:{child_count}children",
            operator.raw(),
            schema
        );
        let pipeline =
            self.close_pipeline_with_source(SchedulerPipelineSource::new(PrototypeSourceNode {
                label: source_label.clone(),
                role,
            }))?;
        let morsel_id = self.alloc_morsel();
        let estimate = estimate_for_leaf(&global_range, schema);
        let priority = MorselPriority::value_work(
            global_range
                .end
                .saturating_sub(global_range.start)
                .max(1)
                .saturating_mul(5),
            estimate.cpu_ns.saturating_add(estimate.io_bytes).max(1),
            0,
        );
        let morsel = SchedulerMorsel::new(
            morsel_id,
            self.domain,
            global_range.clone(),
            role,
            1,
            estimate,
            priority,
        );
        let work = SchedulerWorkTask::new(pipeline, morsel);
        if !self.scheduler.enqueue(SchedulerTask::Work(work), 0) {
            vortex_bail!(
                "layout scheduler queue full while admitting node-output pipeline work for {global_range:?}"
            );
        }
        self.initial_work.push(InitialPipelineWork {
            source_label,
            operator,
            pipeline,
            morsel: morsel_id,
            local_range,
            global_range,
            role,
            schema: schema.to_string(),
        });
        Ok(morsel_id)
    }

    /// Close a pipeline by attaching a source to the currently open pipeline.
    pub(crate) fn close_pipeline_with_source(
        &mut self,
        source: SchedulerPipelineSource,
    ) -> VortexResult<PipelineId> {
        let Some(open_pipeline) = self.open_pipeline.take() else {
            vortex_bail!("cannot close pipeline without an open pipeline");
        };
        let (transforms, sink) = open_pipeline.into_parts();
        Ok(self
            .scheduler
            .close_pipeline_with_source(source, transforms, sink))
    }

    /// Close a pipeline with a segment source and return its
    /// scheduler-local pipeline id.
    pub(crate) fn close_pipeline_with_segment_source(
        &mut self,
        operator: OperatorId,
        segment_id: SegmentId,
        local_range: Range<u64>,
        schema: &DType,
    ) -> VortexResult<PipelineId> {
        let global_range = self.current_global_range.clone();
        self.close_pipeline_with_source(SchedulerPipelineSource::new(PrototypeSourceNode {
            label: format!(
                "operator{}:segment:{segment_id:?}:{local_range:?}->{global_range:?}:{}",
                operator.raw(),
                schema
            ),
            role: role_for_schema(schema),
        }))
    }

    /// Enqueue work for an already-closed pipeline.
    pub(crate) fn enqueue_pipeline_work(
        &mut self,
        pipeline: PipelineId,
        global_range: Range<u64>,
        schema: &DType,
    ) -> VortexResult<MorselId> {
        if pipeline.index() >= self.scheduler.pipeline_count() {
            vortex_bail!(
                "work task referenced unknown pipeline index {}",
                pipeline.index()
            );
        }
        let role = role_for_schema(schema);
        let morsel_id = self.alloc_morsel();
        let estimate = estimate_for_leaf(&global_range, schema);
        let priority = priority_for_leaf(role, &global_range, estimate);
        let stage_count = 1;
        let morsel = SchedulerMorsel::new(
            morsel_id,
            self.domain,
            global_range,
            role,
            stage_count,
            estimate,
            priority,
        );
        let work = SchedulerWorkTask::new(pipeline, morsel);
        if !self.scheduler.enqueue(SchedulerTask::Work(work), 0) {
            vortex_bail!(
                "layout scheduler queue full while registering pipeline work for {pipeline:?}"
            );
        }
        Ok(morsel_id)
    }

    /// Register a segment future with the scheduler.
    pub(crate) fn register_segment_task(
        &mut self,
        pipeline: PipelineId,
        segment_id: SegmentId,
        range: Range<u64>,
        bytes: u64,
        segment_future: SharedSegmentFuture,
    ) -> VortexResult<IoRequestId> {
        if pipeline.index() >= self.scheduler.pipeline_count() {
            vortex_bail!(
                "segment task referenced unknown pipeline index {}",
                pipeline.index()
            );
        }
        let request_id = self.alloc_io_request();
        let rows = range.end.saturating_sub(range.start).max(1);
        let priority = MorselPriority::value_work(rows.saturating_mul(10), bytes.max(1), 0);
        let task = SchedulerSegmentTask::new(
            request_id,
            pipeline,
            segment_id,
            self.domain,
            range.clone(),
            bytes,
            priority,
            segment_future,
        );
        if !self.scheduler.enqueue(SchedulerTask::Segment(task), 0) {
            vortex_bail!(
                "layout scheduler queue full while registering segment work for {range:?}"
            );
        }
        Ok(request_id)
    }

    /// Number of initial scheduler work items emitted by closing
    /// lowered pipelines.
    pub fn initial_work_count(&self) -> usize {
        self.initial_work.len()
    }

    /// Number of queued scheduler events.
    pub fn queued_event_count(&self) -> usize {
        self.scheduler.len()
    }

    /// Number of closed pipelines.
    pub fn pipeline_count(&self) -> usize {
        self.scheduler.pipeline_count()
    }

    /// Queued scheduler memory estimate.
    pub fn queued_memory_bytes(&self) -> u64 {
        self.scheduler.queued_memory_bytes()
    }

    pub(crate) fn initial_work(&self) -> &[InitialPipelineWork] {
        &self.initial_work
    }

    pub(crate) fn pipeline_transforms(
        &self,
        pipeline: PipelineId,
    ) -> Option<&[SchedulerPipelineTransform]> {
        self.scheduler
            .pipeline_state(pipeline)
            .map(|state| state.transforms())
    }

    pub(crate) fn pipeline_sink(&self, pipeline: PipelineId) -> Option<&SchedulerPipelineSink> {
        self.scheduler
            .pipeline_state(pipeline)
            .map(|state| state.sink())
    }

    /// Drive the scheduler until no events remain.
    pub fn drive_to_completion(&mut self) -> LayoutSchedulerRunReport {
        let mut report = LayoutSchedulerRunReport::default();
        let mut tick = 0;
        while let Some(step) = self.scheduler.make_progress(tick) {
            report.steps += 1;
            match step {
                SchedulerStep::Advanced { .. } => report.advanced_morsels += 1,
                SchedulerStep::Completed { .. } => report.completed_morsels += 1,
                SchedulerStep::CompletedSegment { .. } => report.completed_segments += 1,
                SchedulerStep::PendingSegment { .. } => {
                    report.pending_segments += 1;
                    break;
                }
                SchedulerStep::Control { .. } => report.control_events += 1,
            }
            tick = tick.saturating_add(1);
        }
        report
    }

    pub(crate) fn drain_steps(&mut self) -> Vec<SchedulerStep> {
        let mut steps = Vec::new();
        let mut tick = 0;
        while let Some(step) = self.scheduler.make_progress(tick) {
            steps.push(step);
            tick = tick.saturating_add(1);
        }
        steps
    }

    /// Allocate a stable id for a lowered operator/source within this
    /// scheduler instance.
    pub(crate) fn alloc_operator(&mut self) -> OperatorId {
        let id = OperatorId::new(self.next_operator);
        self.next_operator = self.next_operator.saturating_add(1);
        id
    }

    fn alloc_morsel(&mut self) -> MorselId {
        let id = MorselId::new(self.next_morsel);
        self.next_morsel = self.next_morsel.saturating_add(1);
        id
    }

    fn alloc_io_request(&mut self) -> IoRequestId {
        let id = IoRequestId::new(self.next_io_request);
        self.next_io_request = self.next_io_request.saturating_add(1);
        id
    }
}

fn role_for_schema(schema: &DType) -> MorselRole {
    if matches!(schema, DType::Bool(_)) {
        MorselRole::InformationProducer
    } else {
        MorselRole::ValueProducer
    }
}

fn estimate_for_leaf(range: &Range<u64>, schema: &DType) -> MorselEstimate {
    let rows = range.end.saturating_sub(range.start).max(1);
    let bytes_per_row = match schema {
        DType::Bool(_) => 1,
        DType::Primitive(..) => 8,
        DType::Utf8(_) | DType::Binary(_) => 32,
        DType::Struct(..) => 64,
        _ => 16,
    };
    let io_bytes = rows.saturating_mul(bytes_per_row);
    MorselEstimate::new(rows.saturating_mul(10), io_bytes, io_bytes.min(1024 * 1024))
}

fn priority_for_leaf(
    role: MorselRole,
    range: &Range<u64>,
    estimate: MorselEstimate,
) -> MorselPriority {
    let rows = range.end.saturating_sub(range.start).max(1);
    match role {
        MorselRole::InformationProducer => MorselPriority::information_producer(
            rows.saturating_mul(100),
            estimate.cpu_ns.saturating_add(estimate.io_bytes).max(1),
            10_000,
        ),
        _ => MorselPriority::value_work(
            rows.saturating_mul(10),
            estimate.cpu_ns.saturating_add(estimate.io_bytes).max(1),
            0,
        ),
    }
}

fn stage_count_for_role(role: MorselRole) -> u16 {
    match role {
        MorselRole::InformationProducer => 3,
        MorselRole::InformationConsumer
        | MorselRole::ValueProducer
        | MorselRole::Combiner
        | MorselRole::Sink => 2,
    }
}
