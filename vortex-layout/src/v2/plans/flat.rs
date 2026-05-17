// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`FlatPlan`] — terminal plan node over a single segment.
//!
//! Owns its own segment fetch + decode + expression apply; does not
//! go through any V1 `LayoutReader`. Holds the `SegmentId`, the
//! decode context, a pre-registered shared segment future, and the
//! expression — the plan is fully lowered V2-native.
//!
//! When [`LayoutPlan::try_pushdown_mask`] is called and a mask
//! pushes down successfully, this plan node is replaced with a
//! [`crate::v2::plans::filtered_flat::FilteredFlatPlan`] — keeping the two
//! shapes as separate types avoids `Option<mask>` branching on
//! every method.
//!
//! See `LAYOUT_PLAN.md` § Per-layout `plan` walkthrough / `FlatLayout::plan`.

use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;

use async_stream::try_stream;
use futures::FutureExt;
use futures::TryFutureExt;
use futures::future::BoxFuture;
use futures::future::Shared;
use vortex_array::ArrayRef;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::serde::SerializedArray;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_buffer::ByteBuffer;
use vortex_error::SharedVortexResult;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_scan::selection::Selection;
use vortex_session::registry::ReadContext;

use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::v2::demand::RowDemand;
use crate::v2::experiment::trace_flow;
use crate::v2::placeholder::default_array;
use crate::v2::plans::LayoutPlan;
use crate::v2::plans::LayoutPlanRef;
use crate::v2::plans::PartitionStats;
use crate::v2::plans::filtered_flat::FilteredFlatPlan;
use crate::v2::scan_ctx::ScanCtx;
use crate::v2::scheduler::LayoutLoweringCtx;
use crate::v2::scheduler::OutputEstimate;
use crate::v2::scheduler::OutputFrontier;

pub(crate) type SharedSegmentFuture = Shared<BoxFuture<'static, SharedVortexResult<BufferHandle>>>;

/// Terminal node over one segment. Owns the segment fetch and array
/// decode; no V1 `LayoutReader` involved.
pub struct FlatPlan {
    segment_id: SegmentId,
    /// Row count of the underlying segment. The execute `row_range`
    /// is checked against this and a sub-slice is taken after decode.
    layout_row_count: u64,
    /// Un-projected dtype the segment decodes to. The expression is
    /// applied on top to produce `output_dtype`.
    layout_dtype: DType,
    array_ctx: ReadContext,
    /// Optional pre-stored encoding tree (from layout metadata) — when
    /// present, decode reads only the segment buffers and reconstructs
    /// the array via `SerializedArray::from_flatbuffer_and_segment`.
    array_tree: Option<ByteBuffer>,
    segment_fut: SharedSegmentFuture,
    expr: Expression,
    selection: Selection,
    output_dtype: DType,
}

impl FlatPlan {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        segment_id: SegmentId,
        layout_row_count: u64,
        layout_dtype: DType,
        array_ctx: ReadContext,
        array_tree: Option<ByteBuffer>,
        segment_source: Arc<dyn SegmentSource>,
        expr: Expression,
        selection: Selection,
        output_dtype: DType,
    ) -> Self {
        let segment_fut = segment_source
            .request(segment_id)
            .map_err(Arc::new)
            .boxed()
            .shared();
        Self {
            segment_id,
            layout_row_count,
            layout_dtype,
            array_ctx,
            array_tree,
            segment_fut,
            expr,
            selection,
            output_dtype,
        }
    }

    /// Identity tag used by CSE: two `FlatPlan`s with the same
    /// segment ID are reading the same on-disk bytes.
    pub fn segment_id(&self) -> SegmentId {
        self.segment_id
    }

    pub fn expr(&self) -> &Expression {
        &self.expr
    }

    pub fn selection(&self) -> &Selection {
        &self.selection
    }
}

impl PartialEq for FlatPlan {
    fn eq(&self, other: &Self) -> bool {
        // `segment_source` and `array_ctx` are tied to the file we're
        // scanning, not to the plan's identity — two plans within one
        // scan share both, two plans across scans never compare. Skip.
        self.segment_id == other.segment_id
            && self.layout_row_count == other.layout_row_count
            && self.layout_dtype == other.layout_dtype
            && self.array_tree == other.array_tree
            && self.expr == other.expr
            && matches!(
                (&self.selection, &other.selection),
                (Selection::All, Selection::All)
            )
            && self.output_dtype == other.output_dtype
    }
}

impl Eq for FlatPlan {}

impl Hash for FlatPlan {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.segment_id.hash(state);
        self.layout_row_count.hash(state);
        self.layout_dtype.hash(state);
        self.array_tree.hash(state);
        self.expr.hash(state);
        // Today `FlatPlan::execute` only accepts `Selection::All`,
        // so we tag the variant rather than hashing its contents. If
        // we ever support sliced selections we'll need to extend this.
        match self.selection {
            Selection::All => state.write_u8(0),
            _ => state.write_u8(1),
        }
        self.output_dtype.hash(state);
    }
}

impl LayoutPlan for FlatPlan {
    fn schema(&self) -> &DType {
        &self.output_dtype
    }

    fn partition_count(&self) -> usize {
        1
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        if partition >= 1 {
            vortex_bail!("FlatPlan partition out of range: {partition}");
        }
        Ok(PartitionStats::for_range(0..self.layout_row_count))
    }

    fn output_ordered(&self) -> bool {
        true
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        vec![]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![]
    }

    fn children(&self) -> &[LayoutPlanRef] {
        &[]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        if !children.is_empty() {
            vortex_bail!("FlatPlan has no children");
        }
        Ok(self)
    }

    fn try_pushdown_mask(self: Arc<Self>, mask_plan: LayoutPlanRef) -> Option<LayoutPlanRef> {
        if !matches!(mask_plan.schema(), DType::Bool(_)) {
            if trace_flow() {
                tracing::debug!(
                    target: "vortex_layout::v2::flow",
                    segment_id = ?self.segment_id,
                    "flat pushdown failed non-bool mask"
                );
            }
            return None;
        }
        if trace_flow() {
            tracing::debug!(
                target: "vortex_layout::v2::flow",
                segment_id = ?self.segment_id,
                layout_rows = self.layout_row_count,
                output_dtype = %self.output_dtype,
                "flat pushdown succeeded"
            );
        }
        Some(Arc::new(FilteredFlatPlan::with_segment_future(
            self.segment_id,
            self.layout_row_count,
            self.layout_dtype.clone(),
            self.array_ctx.clone(),
            self.array_tree.clone(),
            self.segment_fut.clone(),
            self.expr.clone(),
            self.selection.clone(),
            self.output_dtype.clone(),
            mask_plan,
        )))
    }

    fn lower_to_scheduler(
        &self,
        row_range: Range<u64>,
        ctx: &mut LayoutLoweringCtx,
    ) -> VortexResult<()> {
        if row_range.start > self.layout_row_count || row_range.end > self.layout_row_count {
            vortex_bail!(
                "FlatPlan::lower_to_scheduler row range {row_range:?} exceeds layout row count {}",
                self.layout_row_count
            );
        }
        let subplan = ctx.register_plan_node(row_range.clone(), self.schema(), 0);
        let pipeline = ctx.close_pipeline_with_segment_source(
            subplan,
            self.segment_id,
            row_range,
            self.schema(),
        );
        let global_range = ctx.current_global_range();
        let rows = global_range.end.saturating_sub(global_range.start).max(1);
        ctx.register_segment_task(
            pipeline,
            self.segment_id,
            global_range,
            rows.saturating_mul(16),
            self.segment_fut.clone(),
        )?;
        Ok(())
    }

    fn execute(
        &self,
        row_range: Range<u64>,
        demand: &RowDemand,
        frontier: &OutputFrontier,

        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        if !matches!(self.selection, Selection::All) {
            // The V2 entrypoints never hand FlatPlan a non-`All`
            // selection — `FilterPlan` carries masks separately.
            vortex_bail!("FlatPlan only supports Selection::All in the projection-only path");
        }
        if row_range.start > self.layout_row_count || row_range.end > self.layout_row_count {
            vortex_bail!(
                "FlatPlan::execute row range {row_range:?} exceeds layout row count {}",
                self.layout_row_count
            );
        }

        let dtype = self.output_dtype.clone();
        let output_dtype = dtype.clone();
        let layout_dtype = self.layout_dtype.clone();
        let array_ctx = self.array_ctx.clone();
        let array_tree = self.array_tree.clone();
        let segment_fut = self.segment_fut.clone();
        let layout_row_count = self.layout_row_count;
        let expr = self.expr.clone();
        let session = ctx.session().clone();
        let row_range_for_slice = row_range;
        let demand = demand.clone();
        let mut frontier = frontier.clone();
        if trace_flow() {
            tracing::debug!(
                target: "vortex_layout::v2::flow",
                segment_id = ?self.segment_id,
                row_start = row_range_for_slice.start,
                row_end = row_range_for_slice.end,
                layout_rows = self.layout_row_count,
                output_dtype = %self.output_dtype,
                "flat execute"
            );
        }

        let inner = try_stream! {
            let requested_rows = row_range_for_slice.end - row_range_for_slice.start;
            let mut decoded: Option<ArrayRef> = None;
            let mut cursor = 0;
            while cursor < requested_rows {
                let remaining = requested_rows - cursor;
                let grant = frontier
                    .grant_next_async(remaining, OutputEstimate::new(remaining, remaining))
                    .await?;
                if grant.range().start >= grant.range().end {
                    continue;
                }
                let grant_len = grant.range().end - grant.range().start;
                let local_range = (row_range_for_slice.start + grant.range().start)
                    ..(row_range_for_slice.start + grant.range().end);
                let demanded_rows = demand.cardinality(local_range.clone()).await?;
                if demanded_rows == 0 {
                    let len = usize::try_from(grant_len).map_err(|_| {
                        vortex_err!("FlatPlan: grant length exceeds usize: {grant_len}")
                    })?;
                    yield default_array(&output_dtype, len);
                    cursor = grant.range().end;
                    continue;
                }
                if decoded.is_none() {
                    decoded = Some(decode_segment(
                        segment_fut.clone(),
                        array_tree.clone(),
                        layout_dtype.clone(),
                        layout_row_count,
                        array_ctx.clone(),
                        &session,
                    )
                    .await?);
                }
                let array = decoded
                    .as_ref()
                    .ok_or_else(|| vortex_err!("FlatPlan decoded array missing"))?
                    .clone();
                let array = slice_to_range(array, &local_range)?;
                yield array.apply(&expr)?;
                cursor = grant.range().end;
            }
        };
        Ok(Box::pin(ArrayStreamAdapter::new(dtype, inner)))
    }
}

/// Fetch the segment, parse the array tree (either from layout
/// metadata or from the segment itself), and decode into an
/// `ArrayRef` of length `layout_row_count`. Shared between `FlatPlan`
/// and `FilteredFlatPlan`.
pub(crate) async fn decode_segment(
    segment_fut: SharedSegmentFuture,
    array_tree: Option<ByteBuffer>,
    dtype: DType,
    layout_row_count: u64,
    array_ctx: ReadContext,
    session: &vortex_session::VortexSession,
) -> VortexResult<ArrayRef> {
    let row_count_usize = usize::try_from(layout_row_count)
        .map_err(|_| vortex_err!("FlatPlan: layout row count exceeds usize"))?;
    let segment = segment_fut.await.map_err(VortexError::from)?;
    let parts = if let Some(tree) = array_tree {
        SerializedArray::from_flatbuffer_and_segment(tree, segment)?
    } else {
        SerializedArray::try_from(segment)?
    };
    parts.decode(&dtype, row_count_usize, &array_ctx, session)
}

/// Slice `array` to `row_range` if it doesn't already cover the full
/// length. Cheap when called with `0..len`.
pub(crate) fn slice_to_range(array: ArrayRef, row_range: &Range<u64>) -> VortexResult<ArrayRef> {
    let start = usize::try_from(row_range.start)
        .map_err(|_| vortex_err!("FlatPlan: row range start exceeds usize"))?;
    let end = usize::try_from(row_range.end)
        .map_err(|_| vortex_err!("FlatPlan: row range end exceeds usize"))?;
    if start == 0 && end == array.len() {
        return Ok(array);
    }
    array.slice(start..end)
}
