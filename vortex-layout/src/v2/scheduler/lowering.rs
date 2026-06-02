// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Layout-plan lowering for the single-scheduler prototype.

#![allow(clippy::cognitive_complexity)]

use std::ops::Range;
use std::task::Context;
use std::task::Poll;

use async_stream::try_stream;
use futures::FutureExt;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;

use crate::segments::SegmentId;
use crate::v2::demand::RowDemand;
use crate::v2::domain::DomainId;
use crate::v2::domain::OperatorId;
use crate::v2::experiment::trace_flow;
use crate::v2::plans::LayoutPlanRef;
use crate::v2::plans::flat::SharedSegmentRequest;
use crate::v2::scan_ctx::ScanCtx;
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

    fn can_execute_morsels(&self) -> bool {
        true
    }

    fn push_morsel(
        &self,
        array: ArrayRef,
        ctx: SchedulerRunCtx,
    ) -> futures::future::BoxFuture<'static, VortexResult<()>> {
        async move { ctx.send_root_output(array).await }.boxed()
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
    registered_segments: usize,
    skipped_segments: usize,
    awaited_segments: usize,
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

    /// Number of segment requests admitted after demand checks.
    pub fn registered_segments(&self) -> usize {
        self.registered_segments
    }

    /// Number of segment requests skipped because row demand was
    /// already false for the whole request range.
    pub fn skipped_segments(&self) -> usize {
        self.skipped_segments
    }

    /// Number of segment futures awaited to completion because no
    /// other scheduler work could make progress.
    pub fn awaited_segments(&self) -> usize {
        self.awaited_segments
    }
}

#[derive(Debug, Default)]
struct PipelineSupportSummary {
    unsupported_sources: usize,
    unsupported_transforms: usize,
    unsupported_sinks: usize,
    examples: Vec<String>,
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
        let global_range = self.current_global_range.clone();
        let sink = ResourceSinkNode {
            label: format!(
                "resource:operator{}:input{input}:{local_range:?}->{global_range:?}:{}",
                operator.raw(),
                schema
            ),
        };
        self.with_sink_pipeline(sink, f)
    }

    /// Run `f` with a custom sink open. Multi-input plan nodes use
    /// this to create resource sinks that publish side-input morsels
    /// into state owned by the plan node.
    pub(crate) fn with_sink_pipeline<R>(
        &mut self,
        sink: impl SchedulerSinkNode,
        f: impl FnOnce(&mut Self) -> VortexResult<R>,
    ) -> VortexResult<R> {
        let previous = self.open_pipeline.take();
        self.open_pipeline = Some(OpenPipeline::new(SchedulerPipelineSink::new(sink)));

        let result = f(self);
        if result.is_ok() && self.open_pipeline.is_some() {
            self.open_pipeline = previous;
            vortex_bail!("custom sink pipeline was not closed");
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

    /// Close a pipeline by attaching a source node to the currently
    /// open pipeline.
    pub(crate) fn close_pipeline_with_source_node(
        &mut self,
        source: impl SchedulerSourceNode,
    ) -> VortexResult<PipelineId> {
        self.close_pipeline_with_source(SchedulerPipelineSource::new(source))
    }

    fn close_pipeline_with_source(
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
        self.enqueue_pipeline_work_with_estimate(
            pipeline,
            global_range.clone(),
            role_for_schema(schema),
            estimate_for_leaf(&global_range, schema),
        )
    }

    pub(crate) fn create_pipeline_work_with_estimate(
        &mut self,
        pipeline: PipelineId,
        global_range: Range<u64>,
        role: MorselRole,
        estimate: MorselEstimate,
    ) -> VortexResult<SchedulerWorkTask> {
        if pipeline.index() >= self.scheduler.pipeline_count() {
            vortex_bail!(
                "work task referenced unknown pipeline index {}",
                pipeline.index()
            );
        }
        let morsel_id = self.alloc_morsel();
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
        Ok(SchedulerWorkTask::new(pipeline, morsel))
    }

    fn enqueue_pipeline_work_with_estimate(
        &mut self,
        pipeline: PipelineId,
        global_range: Range<u64>,
        role: MorselRole,
        estimate: MorselEstimate,
    ) -> VortexResult<MorselId> {
        let work =
            self.create_pipeline_work_with_estimate(pipeline, global_range, role, estimate)?;
        let morsel_id = work.morsel().id();
        if !self.scheduler.enqueue(SchedulerTask::Work(work), 0) {
            vortex_bail!(
                "layout scheduler queue full while registering pipeline work for {pipeline:?}"
            );
        }
        Ok(morsel_id)
    }

    /// Register a segment request with the scheduler.
    pub(crate) fn register_segment_task(
        &mut self,
        pipeline: PipelineId,
        segment_id: SegmentId,
        range: Range<u64>,
        bytes: u64,
        segment_request: SharedSegmentRequest,
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
            segment_request,
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

    fn can_execute_morsel_pipelines(&self) -> bool {
        (0..self.scheduler.pipeline_count()).all(|pipeline| {
            let Some(state) = self.scheduler.pipeline_state(PipelineId::new(pipeline)) else {
                return false;
            };
            state.source().can_execute_morsels()
                && state
                    .transforms()
                    .iter()
                    .all(SchedulerPipelineTransform::can_execute_morsels)
                && state.sink().can_execute_morsels()
        })
    }

    fn unsupported_pipeline_summary(&self) -> PipelineSupportSummary {
        let mut summary = PipelineSupportSummary::default();
        for pipeline in 0..self.scheduler.pipeline_count() {
            let Some(state) = self.scheduler.pipeline_state(PipelineId::new(pipeline)) else {
                continue;
            };
            if !state.source().can_execute_morsels() {
                summary.unsupported_sources += 1;
                push_support_example(
                    &mut summary.examples,
                    format!("pipeline{pipeline}:source:{}", state.source().label()),
                );
            }
            for (transform_idx, transform) in state.transforms().iter().enumerate() {
                if !transform.can_execute_morsels() {
                    summary.unsupported_transforms += 1;
                    push_support_example(
                        &mut summary.examples,
                        format!(
                            "pipeline{pipeline}:transform{transform_idx}:{}",
                            transform.label()
                        ),
                    );
                }
            }
            if !state.sink().can_execute_morsels() {
                summary.unsupported_sinks += 1;
                push_support_example(
                    &mut summary.examples,
                    format!("pipeline{pipeline}:sink:{}", state.sink().label()),
                );
            }
        }
        summary
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

    async fn drive_morsel_pipelines(
        mut self,
        run_ctx: SchedulerRunCtx,
    ) -> VortexResult<LayoutSchedulerRunReport> {
        let mut report = LayoutSchedulerRunReport::default();
        let mut tick = 0;
        let trace = trace_flow();
        if trace {
            tracing::debug!(
                target: "vortex_layout::v2::flow",
                scheduler_id = self.scheduler.id().raw(),
                pipelines = self.scheduler.pipeline_count(),
                queued_events = self.scheduler.len(),
                queued_memory_bytes = self.scheduler.queued_memory_bytes(),
                "scheduler morsel driver start"
            );
        }

        while let Some(task) = self.scheduler.pop_task(tick) {
            report.steps += 1;
            match task {
                SchedulerTask::Segment(mut segment) => {
                    if !segment.request_registered() {
                        let pipeline = segment.pipeline;
                        let range = segment.range.clone();
                        let rows = range.end.saturating_sub(range.start).max(1);
                        let role = self
                            .scheduler
                            .pipeline_source(pipeline)
                            .map(SchedulerPipelineSource::role)
                            .ok_or_else(|| {
                                vortex_error::vortex_err!(
                                    "segment task referenced unknown pipeline {pipeline:?}"
                                )
                            })?;
                        match segment.demand_admission(run_ctx.demand()).await? {
                            SegmentDemandAdmission::Skip => {
                                report.skipped_segments += 1;
                                let estimate = MorselEstimate::new(rows.saturating_mul(2), 0, 0);
                                self.enqueue_pipeline_work_with_estimate(
                                    pipeline, range, role, estimate,
                                )?;
                                if trace {
                                    tracing::debug!(
                                        target: "vortex_layout::v2::flow",
                                        scheduler_id = self.scheduler.id().raw(),
                                        tick,
                                        pipeline = pipeline.index(),
                                        request_id = segment.id.raw(),
                                        segment_id = ?segment.segment_id,
                                        "scheduler segment skipped by demand"
                                    );
                                }
                                tick = tick.saturating_add(1);
                                continue;
                            }
                            SegmentDemandAdmission::Register => {
                                report.registered_segments += 1;
                                segment.register_request();
                                if !self
                                    .scheduler
                                    .enqueue(SchedulerTask::Segment(segment), tick)
                                {
                                    vortex_bail!(
                                        "layout scheduler queue full while requeueing registered segment"
                                    );
                                }
                                tick = tick.saturating_add(1);
                                continue;
                            }
                        }
                    }

                    let bytes = if self.scheduler.has_unregistered_segment_request()
                        || self.scheduler.has_non_segment_work()
                    {
                        let waker = futures::task::noop_waker();
                        let mut cx = Context::from_waker(&waker);
                        match segment.poll_once(&mut cx) {
                            Poll::Ready(bytes) => bytes?,
                            Poll::Pending => {
                                report.pending_segments += 1;
                                if !self
                                    .scheduler
                                    .enqueue(SchedulerTask::Segment(segment), tick)
                                {
                                    vortex_bail!(
                                        "layout scheduler queue full while requeueing pending segment"
                                    );
                                }
                                tick = tick.saturating_add(1);
                                continue;
                            }
                        }
                    } else {
                        report.awaited_segments += 1;
                        segment.wait().await?
                    };
                    report.completed_segments += 1;
                    let pipeline = segment.pipeline;
                    let range = segment.range.clone();
                    let rows = range.end.saturating_sub(range.start).max(1);
                    let role = self
                        .scheduler
                        .pipeline_source(pipeline)
                        .map(SchedulerPipelineSource::role)
                        .ok_or_else(|| {
                            vortex_error::vortex_err!(
                                "segment task referenced unknown pipeline {pipeline:?}"
                            )
                        })?;
                    let estimate =
                        MorselEstimate::new(rows.saturating_mul(10), 0, bytes.min(1024 * 1024));
                    self.enqueue_pipeline_work_with_estimate(pipeline, range, role, estimate)?;
                    if trace {
                        tracing::debug!(
                            target: "vortex_layout::v2::flow",
                            scheduler_id = self.scheduler.id().raw(),
                            tick,
                            pipeline = pipeline.index(),
                            request_id = segment.id.raw(),
                            segment_id = ?segment.segment_id,
                            bytes,
                            "scheduler segment ready"
                        );
                    }
                }
                SchedulerTask::Work(work) => {
                    let Some(pipeline_state) =
                        self.scheduler.pipeline_state(work.pipeline).cloned()
                    else {
                        vortex_bail!("work task referenced unknown pipeline {:?}", work.pipeline);
                    };
                    let mut array = pipeline_state
                        .source()
                        .execute_morsel(work.morsel.clone(), run_ctx.clone())
                        .await?;
                    for transform in pipeline_state.transforms() {
                        array = transform.execute_morsel(array, run_ctx.clone()).await?;
                    }
                    pipeline_state
                        .sink()
                        .push_morsel(array, run_ctx.clone())
                        .await?;
                    report.completed_morsels += 1;
                    if trace {
                        tracing::debug!(
                            target: "vortex_layout::v2::flow",
                            scheduler_id = self.scheduler.id().raw(),
                            tick,
                            pipeline = work.pipeline.index(),
                            morsel_id = work.morsel.id.raw(),
                            row_start = work.morsel.order_key.start,
                            row_end = work.morsel.order_key.end,
                            "scheduler morsel done"
                        );
                    }
                }
                SchedulerTask::Control(_) => report.control_events += 1,
            }
            for work in run_ctx.drain_emitted_work()? {
                if !self.scheduler.enqueue(SchedulerTask::Work(work), tick) {
                    vortex_bail!("layout scheduler queue full while enqueueing emitted work");
                }
            }
            tick = tick.saturating_add(1);
        }

        if trace {
            tracing::debug!(
                target: "vortex_layout::v2::flow",
                scheduler_id = self.scheduler.id().raw(),
                steps = report.steps,
                completed_morsels = report.completed_morsels,
                completed_segments = report.completed_segments,
                pending_segments = report.pending_segments,
                registered_segments = report.registered_segments,
                skipped_segments = report.skipped_segments,
                awaited_segments = report.awaited_segments,
                control_events = report.control_events,
                "scheduler morsel driver done"
            );
        }
        Ok(report)
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

/// Try to execute one plan range with the native morsel scheduler.
///
/// This is deliberately conservative: if lowering produces any
/// source, transform, or sink without a morsel implementation, the
/// caller gets `Ok(None)` and should use the existing executor.
pub(crate) fn try_execute_with_single_scheduler(
    plan: LayoutPlanRef,
    row_range: Range<u64>,
    demand: RowDemand,
    ctx: ScanCtx,
) -> VortexResult<Option<SendableArrayStream>> {
    let dtype = plan.schema().clone();
    let mut lowering = LayoutLoweringCtx::for_single_scheduler(row_range.end);
    lowering.open_root_pipeline(row_range.clone(), &dtype);
    if let Err(err) = lowering.with_global_range(row_range.clone(), |lowering| {
        plan.lower_to_scheduler(row_range.clone(), lowering)
    }) {
        tracing::debug!(
            target: "vortex_layout::v2::scheduler",
            row_start = row_range.start,
            row_end = row_range.end,
            dtype = %dtype,
            error = %err,
            "scheduler fallback: lowering unsupported"
        );
        if trace_flow() {
            tracing::debug!(
                target: "vortex_layout::v2::flow",
                error = %err,
                "scheduler morsel lowering unsupported"
            );
        }
        return Ok(None);
    }

    if !lowering.can_execute_morsel_pipelines() {
        let support = lowering.unsupported_pipeline_summary();
        tracing::debug!(
            target: "vortex_layout::v2::scheduler",
            row_start = row_range.start,
            row_end = row_range.end,
            dtype = %dtype,
            pipelines = lowering.pipeline_count(),
            unsupported_sources = support.unsupported_sources,
            unsupported_transforms = support.unsupported_transforms,
            unsupported_sinks = support.unsupported_sinks,
            examples = ?support.examples,
            "scheduler fallback: lowered plan has non-native pipeline nodes"
        );
        return Ok(None);
    }

    let (sink_tx, sink_rx) = kanal::bounded_async::<VortexResult<ArrayRef>>(2);
    let driver_tx = sink_tx;
    let run_ctx = SchedulerRunCtx::new(demand, ctx.clone(), Some(driver_tx.clone()));
    let trace = trace_flow();
    let pipeline_count = lowering.pipeline_count();
    let queued_events = lowering.queued_event_count();
    let queued_memory_bytes = lowering.queued_memory_bytes();
    let row_start = row_range.start;
    let row_end = row_range.end;
    let dtype_label = dtype.to_string();
    tracing::debug!(
        target: "vortex_layout::v2::scheduler",
        row_start,
        row_end,
        pipelines = pipeline_count,
        queued_events,
        queued_memory_bytes,
        dtype = %dtype_label,
        "scheduler run registered"
    );
    if trace {
        tracing::debug!(
            target: "vortex_layout::v2::flow",
            row_start,
            row_end,
            pipelines = pipeline_count,
            queued_events,
            queued_memory_bytes,
            dtype = %dtype,
            "scheduler morsel execute registered"
        );
    }

    ctx.session()
        .handle()
        .spawn(async move {
            match lowering.drive_morsel_pipelines(run_ctx).await {
                Ok(report) => {
                    tracing::debug!(
                        target: "vortex_layout::v2::scheduler",
                        row_start,
                        row_end,
                        dtype = %dtype_label,
                        steps = report.steps,
                        completed_morsels = report.completed_morsels,
                        completed_segments = report.completed_segments,
                        pending_segments = report.pending_segments,
                        registered_segments = report.registered_segments,
                        skipped_segments = report.skipped_segments,
                        awaited_segments = report.awaited_segments,
                        control_events = report.control_events,
                        "scheduler run done"
                    );
                }
                Err(err) => {
                    tracing::debug!(
                        target: "vortex_layout::v2::scheduler",
                        row_start,
                        row_end,
                        dtype = %dtype_label,
                        error = %err,
                        "scheduler run failed"
                    );
                    drop(driver_tx.send(Err(err)).await);
                }
            }
        })
        .detach();

    let stream = try_stream! {
        while let Ok(item) = sink_rx.recv().await {
            yield item?;
        }
    };
    Ok(Some(Box::pin(ArrayStreamAdapter::new(dtype, stream))))
}

fn role_for_schema(schema: &DType) -> MorselRole {
    if matches!(schema, DType::Bool(_)) {
        MorselRole::InformationProducer
    } else {
        MorselRole::ValueProducer
    }
}

fn push_support_example(examples: &mut Vec<String>, example: String) {
    const MAX_EXAMPLES: usize = 8;
    if examples.len() < MAX_EXAMPLES {
        examples.push(example);
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
