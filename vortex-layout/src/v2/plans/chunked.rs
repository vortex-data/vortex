// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`ChunkedPlan`] — partitioning node over an ordered sequence of
//! child plans. One chunk per child.
//!
//! See `LAYOUT_PLAN.md` § Per-layout `plan` walkthrough / `ChunkedPlan`.

#![allow(clippy::cognitive_complexity)]

use std::collections::BTreeMap;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;
use std::sync::Mutex;

use futures::FutureExt;
use futures::TryStreamExt;
use futures::future::BoxFuture;
use futures::stream;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::v2::demand::RowDemand;
use crate::v2::experiment::trace_flow;
use crate::v2::plans::LayoutPlan;
use crate::v2::plans::LayoutPlanRef;
use crate::v2::plans::PartitionStats;
use crate::v2::plans::mask_slice::MaskSlicePlan;
use crate::v2::scan_ctx::ScanCtx;
use crate::v2::scheduler::LayoutLoweringCtx;
use crate::v2::scheduler::queue::MorselEstimate;
use crate::v2::scheduler::queue::MorselRole;
use crate::v2::scheduler::queue::SchedulerMorsel;
use crate::v2::scheduler::queue::SchedulerRunCtx;
use crate::v2::scheduler::queue::SchedulerSinkNode;
use crate::v2::scheduler::queue::SchedulerSourceNode;
use crate::v2::scheduler::queue::SchedulerWorkTask;

/// Routes one partition per child chunk. `partition_count == children.len()`
/// in the default (ordered) mode; relaxed mode is a follow-up PR.
pub struct ChunkedPlan {
    children: Vec<LayoutPlanRef>,
    /// Cumulative row offsets, length `children.len() + 1`. Chunk
    /// `i` covers rows `chunk_offsets[i]..chunk_offsets[i + 1]`.
    chunk_offsets: Vec<u64>,
    output_dtype: DType,
    preserve_order: bool,
}

impl ChunkedPlan {
    pub fn new(children: Vec<LayoutPlanRef>, chunk_offsets: Vec<u64>, output_dtype: DType) -> Self {
        debug_assert_eq!(
            chunk_offsets.len(),
            children.len() + 1,
            "ChunkedPlan: chunk_offsets must have len children.len() + 1",
        );
        Self {
            children,
            chunk_offsets,
            output_dtype,
            preserve_order: true,
        }
    }

    /// In-place flip of the order-preservation flag. See `LAYOUT_PLAN.md`
    /// § Ordered vs. relaxed `ChunkedPlan`.
    pub fn with_preserve_order(self: Arc<Self>, preserve: bool) -> Arc<dyn LayoutPlan> {
        Arc::new(Self {
            children: self.children.clone(),
            chunk_offsets: self.chunk_offsets.clone(),
            output_dtype: self.output_dtype.clone(),
            preserve_order: preserve,
        })
    }

    /// Total row count covered by this Chunked.
    fn total_rows(&self) -> u64 {
        *self.chunk_offsets.last().unwrap_or(&0)
    }

    fn chunk_range(&self, idx: usize) -> Range<u64> {
        self.chunk_offsets[idx]..self.chunk_offsets[idx + 1]
    }
}

#[derive(Debug, Default)]
struct ChunkedSchedulerResourceState {
    outputs: BTreeMap<usize, ArrayRef>,
    parent_work: Option<SchedulerWorkTask>,
    emitted_parent: bool,
}

#[derive(Debug)]
struct ChunkedSchedulerResource {
    expected_inputs: usize,
    state: Mutex<ChunkedSchedulerResourceState>,
}

impl ChunkedSchedulerResource {
    fn new(expected_inputs: usize) -> Self {
        Self {
            expected_inputs,
            state: Mutex::default(),
        }
    }

    fn set_parent_work(&self, work: SchedulerWorkTask) -> VortexResult<()> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| vortex_err!("chunked scheduler resource lock poisoned"))?;
        if state.parent_work.replace(work).is_some() {
            vortex_bail!("chunked scheduler resource parent work set twice");
        }
        Ok(())
    }

    fn record_input(
        &self,
        input: usize,
        array: ArrayRef,
        ctx: &SchedulerRunCtx,
    ) -> VortexResult<()> {
        let parent_work = {
            let mut state = self
                .state
                .lock()
                .map_err(|_| vortex_err!("chunked scheduler resource lock poisoned"))?;
            if state.outputs.insert(input, array).is_some() {
                vortex_bail!("chunked scheduler resource input {input} completed twice");
            }
            if state.outputs.len() == self.expected_inputs && !state.emitted_parent {
                state.emitted_parent = true;
                state.parent_work.clone()
            } else {
                None
            }
        };

        if let Some(work) = parent_work {
            ctx.emit_work(work)?;
        }
        Ok(())
    }

    fn take_outputs(&self) -> VortexResult<Vec<ArrayRef>> {
        let mut state = self
            .state
            .lock()
            .map_err(|_| vortex_err!("chunked scheduler resource lock poisoned"))?;
        if state.outputs.len() != self.expected_inputs {
            vortex_bail!(
                "chunked scheduler resource expected {} inputs, got {}",
                self.expected_inputs,
                state.outputs.len()
            );
        }
        let outputs = std::mem::take(&mut state.outputs);
        Ok(outputs.into_values().collect())
    }
}

struct ChunkedInputSinkNode {
    label: String,
    input: usize,
    resource: Arc<ChunkedSchedulerResource>,
}

impl SchedulerSinkNode for ChunkedInputSinkNode {
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
    ) -> BoxFuture<'static, VortexResult<()>> {
        let input = self.input;
        let resource = Arc::clone(&self.resource);
        async move { resource.record_input(input, array, &ctx) }.boxed()
    }
}

struct ChunkedSchedulerSourceNode {
    label: String,
    resource: Arc<ChunkedSchedulerResource>,
    output_dtype: DType,
}

impl SchedulerSourceNode for ChunkedSchedulerSourceNode {
    fn label(&self) -> &str {
        &self.label
    }

    fn role(&self) -> MorselRole {
        MorselRole::Combiner
    }

    fn can_execute_morsels(&self) -> bool {
        true
    }

    fn execute_morsel(
        &self,
        _morsel: SchedulerMorsel,
        _ctx: SchedulerRunCtx,
    ) -> BoxFuture<'static, VortexResult<ArrayRef>> {
        let resource = Arc::clone(&self.resource);
        let output_dtype = self.output_dtype.clone();
        async move {
            let chunks = resource.take_outputs()?;
            Ok(ChunkedArray::try_new(chunks, output_dtype)?.into_array())
        }
        .boxed()
    }
}

impl PartialEq for ChunkedPlan {
    fn eq(&self, other: &Self) -> bool {
        self.chunk_offsets == other.chunk_offsets
            && self.output_dtype == other.output_dtype
            && self.preserve_order == other.preserve_order
            && crate::v2::plans::plan_slices_eq(&self.children, &other.children)
    }
}

impl Eq for ChunkedPlan {}

impl Hash for ChunkedPlan {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.chunk_offsets.hash(state);
        self.output_dtype.hash(state);
        self.preserve_order.hash(state);
        crate::v2::plans::hash_plan_slice(&self.children, state);
    }
}

impl LayoutPlan for ChunkedPlan {
    fn schema(&self) -> &DType {
        &self.output_dtype
    }

    fn partition_count(&self) -> usize {
        self.children.len()
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        if partition >= self.children.len() {
            vortex_bail!("ChunkedPlan partition out of range: {partition}");
        }
        Ok(PartitionStats::for_range(self.chunk_range(partition)))
    }

    fn output_ordered(&self) -> bool {
        self.preserve_order
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        vec![true; self.children.len()]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        // When preserving order, we route partition k → children[k]
        // with no reordering, so each child's order is preserved.
        // When relaxed, we may emit children in arrival order.
        vec![self.preserve_order; self.children.len()]
    }

    fn children(&self) -> &[LayoutPlanRef] {
        &self.children
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        if children.len() != self.children.len() {
            vortex_bail!(
                "ChunkedPlan::with_new_children expected {} children, got {}",
                self.children.len(),
                children.len()
            );
        }
        Ok(Arc::new(Self {
            children,
            chunk_offsets: self.chunk_offsets.clone(),
            output_dtype: self.output_dtype.clone(),
            preserve_order: self.preserve_order,
        }))
    }

    fn try_pushdown_mask(self: Arc<Self>, mask_plan: LayoutPlanRef) -> Option<LayoutPlanRef> {
        let trace = trace_flow();
        if !matches!(mask_plan.schema(), DType::Bool(_)) {
            if trace {
                tracing::debug!(
                    target: "vortex_layout::v2::flow",
                    chunk_count = self.children.len(),
                    "chunked pushdown failed non-bool mask"
                );
            }
            return None;
        }
        // Push the chunk's slice of the mask into each child. Each
        // child sees the mask in its own (chunk-local) row coordinate
        // space via `MaskSlicePlan`. If any child rejects the push,
        // we bail and let the caller wrap with `FilterPlan`.
        //
        // FIXME: each child's `MaskSlicePlan` independently re-executes
        // the upstream mask plan over its slice. For mask plans that
        // do real work (e.g. `ZonedPruningPlan` reads the zones table
        // each call) this is redundant — the right fix is the CSE
        // pass + `Let` / `Use` to share the mask evaluation across
        // chunks. For now the redundancy is honest.
        let mut new_children = Vec::with_capacity(self.children.len());
        for idx in 0..self.children.len() {
            let chunk_start = self.chunk_offsets[idx];
            let chunk_end = self.chunk_offsets[idx + 1];
            let sliced: LayoutPlanRef = Arc::new(MaskSlicePlan::new(
                Arc::clone(&mask_plan),
                chunk_start..chunk_end,
            ));
            let Some(pushed) = Arc::clone(&self.children[idx]).try_pushdown_mask(sliced) else {
                if trace {
                    tracing::debug!(
                        target: "vortex_layout::v2::flow",
                        chunk_idx = idx,
                        chunk_start,
                        chunk_end,
                        chunk_count = self.children.len(),
                        "chunked pushdown failed child rejected"
                    );
                }
                return None;
            };
            if trace {
                tracing::debug!(
                    target: "vortex_layout::v2::flow",
                    chunk_idx = idx,
                    chunk_start,
                    chunk_end,
                    chunk_count = self.children.len(),
                    "chunked pushdown child succeeded"
                );
            }
            new_children.push(pushed);
        }
        if trace {
            tracing::debug!(
                target: "vortex_layout::v2::flow",
                chunk_count = self.children.len(),
                "chunked pushdown succeeded"
            );
        }
        Some(Arc::new(Self {
            children: new_children,
            chunk_offsets: self.chunk_offsets.clone(),
            output_dtype: self.output_dtype.clone(),
            preserve_order: self.preserve_order,
        }))
    }

    fn lower_to_scheduler(
        &self,
        row_range: Range<u64>,
        ctx: &mut LayoutLoweringCtx,
    ) -> VortexResult<()> {
        if row_range.start > self.total_rows() || row_range.end > self.total_rows() {
            vortex_bail!(
                "ChunkedPlan::lower_to_scheduler row range {row_range:?} exceeds total {}",
                self.total_rows()
            );
        }
        if row_range.start > row_range.end {
            vortex_bail!(
                "ChunkedPlan::lower_to_scheduler received reversed row range {row_range:?}"
            );
        }

        let operator = ctx.alloc_operator();
        let mut intersecting = Vec::new();
        for idx in 0..self.children.len() {
            let chunk_start = self.chunk_offsets[idx];
            let chunk_end = self.chunk_offsets[idx + 1];
            if chunk_end <= row_range.start || chunk_start >= row_range.end {
                continue;
            }

            let intersect_start = chunk_start.max(row_range.start);
            let intersect_end = chunk_end.min(row_range.end);
            let child_range = (intersect_start - chunk_start)..(intersect_end - chunk_start);
            intersecting.push((idx, child_range, intersect_start..intersect_end));
        }

        if intersecting.is_empty() {
            ctx.close_node_output_pipeline(operator, row_range, self.schema(), 0)?;
            return Ok(());
        }

        let resource = Arc::new(ChunkedSchedulerResource::new(intersecting.len()));
        for (input, (idx, child_range, global_range)) in intersecting.iter().enumerate() {
            let sink = ChunkedInputSinkNode {
                label: format!(
                    "chunked-resource:operator{}:input{input}:child{idx}:{child_range:?}->{global_range:?}:{}",
                    operator.raw(),
                    self.children[*idx].schema()
                ),
                input,
                resource: Arc::clone(&resource),
            };
            ctx.with_global_range(global_range.clone(), |ctx| {
                ctx.with_sink_pipeline(sink, |ctx| {
                    self.children[*idx].lower_to_scheduler(child_range.clone(), ctx)
                })
            })?;
        }

        let global_range = ctx.current_global_range();
        let source = ChunkedSchedulerSourceNode {
            label: format!(
                "operator{}:chunked-output:{row_range:?}->{global_range:?}:{}:{}children",
                operator.raw(),
                self.schema(),
                intersecting.len()
            ),
            resource: Arc::clone(&resource),
            output_dtype: self.output_dtype.clone(),
        };
        let pipeline = ctx.close_pipeline_with_source_node(source)?;
        let rows = global_range.end.saturating_sub(global_range.start).max(1);
        let estimate = MorselEstimate::new(
            rows.saturating_mul(5),
            0,
            rows.saturating_mul(16).min(1024 * 1024),
        );
        let parent_work = ctx.create_pipeline_work_with_estimate(
            pipeline,
            global_range,
            MorselRole::Combiner,
            estimate,
        )?;
        resource.set_parent_work(parent_work)?;
        Ok(())
    }

    fn execute(
        &self,
        row_range: Range<u64>,
        demand: &RowDemand,

        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        if row_range.start > self.total_rows() || row_range.end > self.total_rows() {
            vortex_bail!(
                "ChunkedPlan::execute row range {row_range:?} exceeds total {}",
                self.total_rows()
            );
        }
        if row_range.start > row_range.end {
            vortex_bail!("ChunkedPlan::execute received reversed row range {row_range:?}");
        }

        // Find chunks intersecting the requested range. Walk each
        // intersecting chunk and dispatch a sub-range relative to
        // the chunk's own row coordinate space, scoping demand the
        // same way.
        //
        // Chunks are ordered and disjoint by construction, so a
        // simple linear scan is fine for typical chunk counts. If a
        // single Chunked grows to thousands of chunks we can swap to
        // binary search over `chunk_offsets`.
        let mut child_streams: Vec<SendableArrayStream> = Vec::new();
        let trace = trace_flow();
        if trace {
            tracing::debug!(
                target: "vortex_layout::v2::flow",
                row_start = row_range.start,
                row_end = row_range.end,
                chunk_count = self.children.len(),
                preserve_order = self.preserve_order,
                "chunked execute"
            );
        }
        for idx in 0..self.children.len() {
            let chunk_start = self.chunk_offsets[idx];
            let chunk_end = self.chunk_offsets[idx + 1];
            if chunk_end <= row_range.start || chunk_start >= row_range.end {
                continue;
            }
            let intersect_start = chunk_start.max(row_range.start);
            let intersect_end = chunk_end.min(row_range.end);
            // Translate to the child's own row coordinates.
            let child_range = (intersect_start - chunk_start)..(intersect_end - chunk_start);
            // Scope demand to the full child chunk so the child's
            // local row coordinates line up with demand's local row
            // coordinates, even when the requested parent range
            // starts in the middle of the chunk.
            let child_demand = demand.scope(chunk_start..chunk_end);
            if trace {
                tracing::debug!(
                    target: "vortex_layout::v2::flow",
                    chunk_idx = idx,
                    chunk_start,
                    chunk_end,
                    child_start = child_range.start,
                    child_end = child_range.end,
                    "chunked child execute"
                );
            }
            child_streams.push(self.children[idx].execute(child_range, &child_demand, ctx)?);
        }

        let dtype = self.output_dtype.clone();
        let flat = stream::iter(child_streams.into_iter().map(VortexResult::Ok)).try_flatten();
        Ok(Box::pin(ArrayStreamAdapter::new(dtype, flat)))
    }
}
