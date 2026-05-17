// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Layout-plan lowering and the current single-scheduler execution bridge.

#![allow(clippy::cognitive_complexity)]

use std::ops::Range;
use std::sync::Arc;
use std::time::Instant;

use async_stream::try_stream;
use futures::StreamExt;
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
use crate::v2::domain::SubplanId;
use crate::v2::experiment::trace_flow;
use crate::v2::plans::LayoutPlanRef;
use crate::v2::plans::flat::SharedSegmentFuture;
use crate::v2::scan_ctx::ScanCtx;
use crate::v2::scheduler::frontier::OutputFrontier;
use crate::v2::scheduler::queue::*;

/// A lowered layout-plan node recorded by the scheduler prototype.
///
/// This is intentionally descriptive metadata, not an executable plan
/// node. The executable unit in this prototype is the scheduler event
/// registered by leaves.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LoweredLayoutNode {
    id: SubplanId,
    local_range: Range<u64>,
    global_range: Range<u64>,
    schema: String,
    child_count: usize,
}

impl LoweredLayoutNode {
    pub(crate) fn id(&self) -> SubplanId {
        self.id
    }

    pub(crate) fn child_count(&self) -> usize {
        self.child_count
    }

    pub(crate) fn global_range(&self) -> &Range<u64> {
        &self.global_range
    }
}

/// One initial leaf work item produced by layout lowering.
#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct LoweredLeafWork {
    subplan: SubplanId,
    pipeline: PipelineId,
    morsel: MorselId,
    local_range: Range<u64>,
    global_range: Range<u64>,
    role: MorselRole,
    schema: String,
}

impl LoweredLeafWork {
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
/// This is the bridge between the recursive [`crate::v2::plans::LayoutPlan`]
/// tree and the scheduler sketch above. Plan nodes record metadata;
/// leaves enqueue initial morsels into one partition-local scheduler.
/// The scheduler can then be driven by repeatedly popping the highest
/// priority event and executing one abstract stage.
pub struct LayoutLoweringCtx {
    scheduler: PartitionScheduler,
    domain: DomainId,
    current_global_range: Range<u64>,
    next_subplan: u32,
    next_morsel: u64,
    next_io_request: u64,
    nodes: Vec<LoweredLayoutNode>,
    leaves: Vec<LoweredLeafWork>,
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
            next_subplan: 1,
            next_morsel: 1,
            next_io_request: 1,
            nodes: Vec::new(),
            leaves: Vec::new(),
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

    /// Record a plan node and return its prototype sub-plan id.
    pub(crate) fn register_plan_node(
        &mut self,
        local_range: Range<u64>,
        schema: &DType,
        child_count: usize,
    ) -> SubplanId {
        let id = self.alloc_subplan();
        self.nodes.push(LoweredLayoutNode {
            id,
            local_range,
            global_range: self.current_global_range.clone(),
            schema: schema.to_string(),
            child_count,
        });
        id
    }

    /// Register initial work for a leaf in the current global range.
    pub(crate) fn register_leaf_work(
        &mut self,
        subplan: SubplanId,
        local_range: Range<u64>,
        schema: &DType,
    ) -> VortexResult<()> {
        let role = role_for_schema(schema);
        let global_range = self.current_global_range.clone();
        let pipeline = self.close_pipeline_with_source(SchedulerPipelineSource::Leaf {
            subplan,
            local_range: local_range.clone(),
            global_range: global_range.clone(),
            schema: schema.to_string(),
            role,
        });
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
                "layout scheduler queue full while registering leaf work for {global_range:?}"
            );
        }

        self.leaves.push(LoweredLeafWork {
            subplan,
            pipeline,
            morsel: morsel_id,
            local_range,
            global_range,
            role,
            schema: schema.to_string(),
        });
        Ok(())
    }

    /// Close a pipeline with an abstract leaf source.
    pub(crate) fn close_pipeline_with_source(
        &mut self,
        source: SchedulerPipelineSource,
    ) -> PipelineId {
        self.scheduler.close_pipeline_with_source(source)
    }

    /// Close a pipeline with a segment source and return its
    /// scheduler-local pipeline id.
    pub(crate) fn close_pipeline_with_segment_source(
        &mut self,
        subplan: SubplanId,
        segment_id: SegmentId,
        local_range: Range<u64>,
        schema: &DType,
    ) -> PipelineId {
        self.close_pipeline_with_source(SchedulerPipelineSource::Segment {
            subplan,
            segment_id,
            local_range,
            global_range: self.current_global_range.clone(),
            schema: schema.to_string(),
        })
    }

    /// Close a pipeline whose source runs an already-built plan into
    /// the scheduler sink. This is the first runnable bridge; finer
    /// lowering can replace it one plan node at a time.
    pub(crate) fn close_pipeline_with_execute_source(
        &mut self,
        plan: LayoutPlanRef,
        row_range: Range<u64>,
        demand: RowDemand,
        frontier: OutputFrontier,
        ctx: ScanCtx,
    ) -> PipelineId {
        self.close_pipeline_with_source(SchedulerPipelineSource::ExecutePlan {
            plan,
            row_range,
            demand,
            frontier,
            ctx,
        })
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

    /// Number of lowered plan nodes.
    pub fn lowered_node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of initial leaf work items.
    pub fn leaf_work_count(&self) -> usize {
        self.leaves.len()
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

    pub(crate) fn lowered_nodes(&self) -> &[LoweredLayoutNode] {
        &self.nodes
    }

    pub(crate) fn leaf_work(&self) -> &[LoweredLeafWork] {
        &self.leaves
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

    async fn drive_to_sink(
        mut self,
        sink: kanal::AsyncSender<VortexResult<ArrayRef>>,
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
                "scheduler driver start"
            );
        }
        while let Some(task) = self.scheduler.pop_task(tick) {
            report.steps += 1;
            match task {
                SchedulerTask::Work(mut work) => {
                    match self.scheduler.pipeline_source(work.pipeline).cloned() {
                        Some(SchedulerPipelineSource::ExecutePlan {
                            plan,
                            row_range,
                            demand,
                            frontier,
                            ctx,
                        }) => {
                            report.completed_morsels += 1;
                            let row_start = row_range.start;
                            let row_end = row_range.end;
                            if trace {
                                tracing::debug!(
                                    target: "vortex_layout::v2::flow",
                                    scheduler_id = self.scheduler.id().raw(),
                                    tick,
                                    pipeline = work.pipeline.index(),
                                    morsel_id = work.morsel.id.raw(),
                                    row_start,
                                    row_end,
                                    rows = row_end.saturating_sub(row_start),
                                    "scheduler execute plan start"
                                );
                            }
                            let execute_start = Instant::now();
                            let mut stream = plan.execute(row_range, &demand, &frontier, &ctx)?;
                            let mut array_idx = 0usize;
                            let mut total_rows = 0usize;
                            loop {
                                let next_start = Instant::now();
                                let Some(item) = stream.next().await else {
                                    break;
                                };
                                let next_elapsed_ms = next_start.elapsed().as_secs_f64() * 1000.0;
                                let array = item?;
                                let rows = array.len();
                                total_rows = total_rows.saturating_add(rows);
                                if trace {
                                    tracing::debug!(
                                        target: "vortex_layout::v2::flow",
                                        scheduler_id = self.scheduler.id().raw(),
                                        tick,
                                        pipeline = work.pipeline.index(),
                                        morsel_id = work.morsel.id.raw(),
                                        array_idx,
                                        rows,
                                        total_rows,
                                        elapsed_ms = next_elapsed_ms,
                                        "scheduler execute stream next"
                                    );
                                }
                                let send_start = Instant::now();
                                if sink.send(Ok(array)).await.is_err() {
                                    if trace {
                                        tracing::debug!(
                                            target: "vortex_layout::v2::flow",
                                            scheduler_id = self.scheduler.id().raw(),
                                            tick,
                                            pipeline = work.pipeline.index(),
                                            morsel_id = work.morsel.id.raw(),
                                            array_idx,
                                            rows,
                                            send_elapsed_ms =
                                                send_start.elapsed().as_secs_f64() * 1000.0,
                                            "scheduler sink closed"
                                        );
                                    }
                                    return Ok(report);
                                }
                                if trace {
                                    tracing::debug!(
                                        target: "vortex_layout::v2::flow",
                                        scheduler_id = self.scheduler.id().raw(),
                                        tick,
                                        pipeline = work.pipeline.index(),
                                        morsel_id = work.morsel.id.raw(),
                                        array_idx,
                                        rows,
                                        send_elapsed_ms =
                                            send_start.elapsed().as_secs_f64() * 1000.0,
                                        "scheduler sink sent"
                                    );
                                }
                                array_idx = array_idx.saturating_add(1);
                            }
                            if trace {
                                tracing::debug!(
                                    target: "vortex_layout::v2::flow",
                                    scheduler_id = self.scheduler.id().raw(),
                                    tick,
                                    pipeline = work.pipeline.index(),
                                    morsel_id = work.morsel.id.raw(),
                                    arrays = array_idx,
                                    rows = total_rows,
                                    elapsed_ms = execute_start.elapsed().as_secs_f64() * 1000.0,
                                    "scheduler execute plan done"
                                );
                            }
                        }
                        _ => {
                            let morsel_id = work.morsel.id;
                            let pipeline = work.pipeline;
                            if work.morsel.advance_one_stage().is_some() {
                                if !self.scheduler.enqueue(SchedulerTask::Work(work), tick) {
                                    vortex_bail!(
                                        "layout scheduler queue full while requeueing work for {pipeline:?}"
                                    );
                                }
                                report.advanced_morsels += 1;
                            } else {
                                let _ = morsel_id;
                                report.completed_morsels += 1;
                            }
                        }
                    }
                }
                SchedulerTask::Segment(mut segment) => {
                    let wait_start = Instant::now();
                    let _bytes = segment.wait().await?;
                    report.completed_segments += 1;
                    if trace {
                        tracing::debug!(
                            target: "vortex_layout::v2::flow",
                            scheduler_id = self.scheduler.id().raw(),
                            tick,
                            pipeline = segment.pipeline.index(),
                            request_id = segment.id.raw(),
                            segment_id = ?segment.segment_id,
                            row_start = segment.range.start,
                            row_end = segment.range.end,
                            rows = segment.range.end.saturating_sub(segment.range.start),
                            bytes = segment.bytes,
                            elapsed_ms = wait_start.elapsed().as_secs_f64() * 1000.0,
                            "scheduler segment done"
                        );
                    }
                    // The next step is to store the segment bytes in
                    // this pipeline's local state and enqueue pipeline
                    // work that decodes and pushes to the sink.
                }
                SchedulerTask::Control(_) => report.control_events += 1,
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
                control_events = report.control_events,
                "scheduler driver done"
            );
        }
        Ok(report)
    }

    fn alloc_subplan(&mut self) -> SubplanId {
        let id = SubplanId::new(self.next_subplan);
        self.next_subplan = self.next_subplan.saturating_add(1);
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

/// Execute one partition by spawning a scheduler driver and returning
/// a stream over its sink queue.
///
/// This is intentionally a compatibility bridge: the root pipeline
/// source delegates to the existing `LayoutPlan::execute` so the
/// scheduler/queue shape can run end-to-end before every plan node has
/// a native pipeline implementation.
pub(crate) fn execute_with_single_scheduler(
    plan: LayoutPlanRef,
    row_range: Range<u64>,
    demand: RowDemand,
    frontier: OutputFrontier,
    ctx: ScanCtx,
) -> VortexResult<SendableArrayStream> {
    let dtype = plan.schema().clone();
    let mut lowering = LayoutLoweringCtx::for_single_scheduler(row_range.end);
    let trace = trace_flow();
    let pipeline = lowering.close_pipeline_with_execute_source(
        Arc::clone(&plan),
        row_range.clone(),
        demand,
        frontier,
        ctx.clone(),
    );
    let morsel_id = lowering.enqueue_pipeline_work(pipeline, row_range.clone(), &dtype)?;

    if trace {
        tracing::debug!(
            target: "vortex_layout::v2::flow",
            pipeline = pipeline.index(),
            morsel_id = morsel_id.raw(),
            row_start = row_range.start,
            row_end = row_range.end,
            rows = row_range.end.saturating_sub(row_range.start),
            pipelines = lowering.pipeline_count(),
            queued_events = lowering.queued_event_count(),
            queued_memory_bytes = lowering.queued_memory_bytes(),
            dtype = %dtype,
            "scheduler execute registered"
        );
    }

    let (sink_tx, sink_rx) = kanal::bounded_async::<VortexResult<ArrayRef>>(2);
    let driver_tx = sink_tx;
    ctx.session()
        .handle()
        .spawn(async move {
            if let Err(err) = lowering.drive_to_sink(driver_tx.clone()).await {
                drop(driver_tx.send(Err(err)).await);
            }
        })
        .detach();

    let stream = try_stream! {
        let mut array_idx = 0usize;
        while let Ok(item) = sink_rx.recv().await {
            if trace {
                match &item {
                    Ok(array) => {
                        tracing::debug!(
                            target: "vortex_layout::v2::flow",
                            array_idx,
                            rows = array.len(),
                            "scheduler sink received"
                        );
                    }
                    Err(err) => {
                        tracing::debug!(
                            target: "vortex_layout::v2::flow",
                            array_idx,
                            error = %err,
                            "scheduler sink received error"
                        );
                    }
                }
            }
            array_idx = array_idx.saturating_add(1);
            yield item?;
        }
    };
    Ok(Box::pin(ArrayStreamAdapter::new(dtype, stream)))
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
