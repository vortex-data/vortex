// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`FlatPlan`] — terminal plan node over a single segment.
//!
//! Owns its own segment fetch + decode + expression apply; does not
//! go through any V1 `LayoutReader`. Holds the `SegmentId`, the
//! decode context, a lazy shared segment request, and the expression
//! — the plan is fully lowered V2-native.
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
use parking_lot::Mutex;
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
use crate::v2::scheduler::queue::MorselRole;
use crate::v2::scheduler::queue::SchedulerMorsel;
use crate::v2::scheduler::queue::SchedulerRunCtx;
use crate::v2::scheduler::queue::SchedulerSourceNode;

pub(crate) type SharedSegmentFuture = Shared<BoxFuture<'static, SharedVortexResult<BufferHandle>>>;

/// Lazily registers a segment read and shares the resulting future.
#[derive(Clone)]
pub(crate) struct SharedSegmentRequest {
    inner: Arc<SharedSegmentRequestInner>,
}

struct SharedSegmentRequestInner {
    segment_id: SegmentId,
    segment_source: Arc<dyn SegmentSource>,
    future: Mutex<Option<SharedSegmentFuture>>,
}

impl SharedSegmentRequest {
    pub(crate) fn new(segment_source: Arc<dyn SegmentSource>, segment_id: SegmentId) -> Self {
        Self {
            inner: Arc::new(SharedSegmentRequestInner {
                segment_id,
                segment_source,
                future: Mutex::new(None),
            }),
        }
    }

    pub(crate) fn request(&self) -> SharedSegmentFuture {
        let mut future = self.inner.future.lock();
        future
            .get_or_insert_with(|| {
                self.inner
                    .segment_source
                    .request(self.inner.segment_id)
                    .map_err(Arc::new)
                    .boxed()
                    .shared()
            })
            .clone()
    }
}

impl std::fmt::Debug for SharedSegmentRequest {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedSegmentRequest")
            .field("segment_id", &self.inner.segment_id)
            .finish_non_exhaustive()
    }
}

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
    segment_request: SharedSegmentRequest,
    expr: Expression,
    selection: Selection,
    output_dtype: DType,
}

struct FlatSchedulerSourceNode {
    label: String,
    segment_id: SegmentId,
    local_range: Range<u64>,
    layout_row_count: u64,
    layout_dtype: DType,
    array_ctx: ReadContext,
    array_tree: Option<ByteBuffer>,
    segment_request: SharedSegmentRequest,
    expr: Expression,
    selection: Selection,
    output_dtype: DType,
}

impl SchedulerSourceNode for FlatSchedulerSourceNode {
    fn label(&self) -> &str {
        &self.label
    }

    fn role(&self) -> MorselRole {
        MorselRole::ValueProducer
    }

    fn can_execute_morsels(&self) -> bool {
        true
    }

    fn execute_morsel(
        &self,
        morsel: SchedulerMorsel,
        ctx: SchedulerRunCtx,
    ) -> BoxFuture<'static, VortexResult<ArrayRef>> {
        let segment_id = self.segment_id;
        let local_range = self.local_range.clone();
        let layout_row_count = self.layout_row_count;
        let layout_dtype = self.layout_dtype.clone();
        let array_ctx = self.array_ctx.clone();
        let array_tree = self.array_tree.clone();
        let segment_request = self.segment_request.clone();
        let expr = self.expr.clone();
        let selection = self.selection.clone();
        let output_dtype = self.output_dtype.clone();
        let global_range = morsel.order_key().clone();
        async move {
            if !matches!(selection, Selection::All) {
                vortex_bail!(
                    "FlatPlan scheduler source only supports Selection::All for {segment_id:?}"
                );
            }

            let requested_rows = local_range.end.saturating_sub(local_range.start);
            let demanded_rows = ctx.demand().cardinality(global_range).await?;
            if demanded_rows == 0 {
                let len = usize::try_from(requested_rows).map_err(|_| {
                    vortex_err!("FlatPlan scheduler source requested row count exceeds usize")
                })?;
                return Ok(default_array(&output_dtype, len));
            }

            let session = ctx.scan_ctx().session().clone();
            let array = decode_segment(
                segment_request.request(),
                array_tree,
                layout_dtype,
                layout_row_count,
                array_ctx,
                &session,
            )
            .await?;
            let array = slice_to_range(array, &local_range)?;
            array.apply(&expr)
        }
        .boxed()
    }
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
        Self {
            segment_id,
            layout_row_count,
            layout_dtype,
            array_ctx,
            array_tree,
            segment_request: SharedSegmentRequest::new(segment_source, segment_id),
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
        Some(Arc::new(FilteredFlatPlan::with_segment_request(
            self.segment_id,
            self.layout_row_count,
            self.layout_dtype.clone(),
            self.array_ctx.clone(),
            self.array_tree.clone(),
            self.segment_request.clone(),
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
        let operator = ctx.alloc_operator();
        let label = format!(
            "operator{}:flat-segment:{:?}:{:?}->{}",
            operator.raw(),
            self.segment_id,
            row_range,
            self.output_dtype
        );
        let pipeline = ctx.close_pipeline_with_source_node(FlatSchedulerSourceNode {
            label,
            segment_id: self.segment_id,
            local_range: row_range,
            layout_row_count: self.layout_row_count,
            layout_dtype: self.layout_dtype.clone(),
            array_ctx: self.array_ctx.clone(),
            array_tree: self.array_tree.clone(),
            segment_request: self.segment_request.clone(),
            expr: self.expr.clone(),
            selection: self.selection.clone(),
            output_dtype: self.output_dtype.clone(),
        })?;
        let global_range = ctx.current_global_range();
        let rows = global_range.end.saturating_sub(global_range.start).max(1);
        ctx.register_segment_task(
            pipeline,
            self.segment_id,
            global_range,
            rows.saturating_mul(16),
            self.segment_request.clone(),
        )?;
        Ok(())
    }

    fn execute(
        &self,
        row_range: Range<u64>,
        demand: &RowDemand,
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
        let segment_request = self.segment_request.clone();
        let layout_row_count = self.layout_row_count;
        let expr = self.expr.clone();
        let session = ctx.session().clone();
        let row_range_for_slice = row_range;
        let demand = demand.clone();
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
            let demanded_rows = demand.cardinality(row_range_for_slice.clone()).await?;
            if demanded_rows == 0 {
                let len = usize::try_from(requested_rows).map_err(|_| {
                    vortex_err!("FlatPlan: requested row count exceeds usize: {requested_rows}")
                })?;
                yield default_array(&output_dtype, len);
            } else {
                let array = decode_segment(
                    segment_request.request(),
                    array_tree.clone(),
                    layout_dtype.clone(),
                    layout_row_count,
                    array_ctx.clone(),
                    &session,
                )
                .await?;
                let array = slice_to_range(array, &row_range_for_slice)?;
                yield array.apply(&expr)?;
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
