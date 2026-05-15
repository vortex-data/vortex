// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`FlatPlan`] — terminal plan node over a single segment.
//!
//! Owns its own segment fetch + decode + expression apply; does not
//! go through any V1 `LayoutReader`. Holds the `SegmentId`, the
//! decode context, the `SegmentSource`, and the expression — the
//! plan is fully lowered V2-native.
//!
//! Optionally absorbs a `mask_plan` (via [`LayoutPlan::try_pushdown_mask`])
//! and applies the resulting mask to the read; downstream
//! `FilterPlan` is dropped on successful pushdown. Today the mask is
//! awaited as a single `Mask` and applied after decode; the
//! sub-segment-aware variant lands later (see `FuseFilterIntoFlat` in
//! `LAYOUT_PLAN.md`).
//!
//! See `LAYOUT_PLAN.md` § Per-layout `plan` walkthrough / `FlatLayout::plan`
//! and § FilterPlan and its pushdown.

use std::ops::Range;
use std::sync::Arc;

use async_stream::try_stream;
use futures::FutureExt;
use futures::stream;
use vortex_array::ArrayRef;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::serde::SerializedArray;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_mask::Mask;
use vortex_scan::selection::Selection;
use vortex_session::registry::ReadContext;

use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;
use crate::v2::scan_ctx::ScanCtx;

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
    segment_source: Arc<dyn SegmentSource>,
    expr: Expression,
    selection: Selection,
    output_dtype: DType,
    /// Optional pushed-down mask. If set, executed at execute-time
    /// over the same row range and its result is folded into a `Mask`
    /// that is `filter`'d against the decoded array before the
    /// expression is applied.
    mask_plan: Option<LayoutPlanRef>,
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
            segment_source,
            expr,
            selection,
            output_dtype,
            mask_plan: None,
        }
    }

    /// Identity tag used by future CSE: two `FlatPlan`s with the same
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
            return None;
        }
        if self.mask_plan.is_some() {
            // Stacked masks would need an AND wrapper; skip for now.
            return None;
        }
        Some(Arc::new(Self {
            segment_id: self.segment_id,
            layout_row_count: self.layout_row_count,
            layout_dtype: self.layout_dtype.clone(),
            array_ctx: self.array_ctx.clone(),
            array_tree: self.array_tree.clone(),
            segment_source: Arc::clone(&self.segment_source),
            expr: self.expr.clone(),
            selection: self.selection.clone(),
            output_dtype: self.output_dtype.clone(),
            mask_plan: Some(mask_plan),
        }))
    }

    fn execute(&self, row_range: Range<u64>, ctx: &ScanCtx) -> VortexResult<SendableArrayStream> {
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
        let layout_dtype = self.layout_dtype.clone();
        let array_ctx = self.array_ctx.clone();
        let array_tree = self.array_tree.clone();
        let segment_source = Arc::clone(&self.segment_source);
        let segment_id = self.segment_id;
        let layout_row_count = self.layout_row_count;
        let expr = self.expr.clone();
        let session = ctx.session().clone();
        let row_range_for_slice = row_range.clone();

        // Fast path: no pushed-down mask. Fetch + decode + slice + apply.
        let Some(mask_plan) = &self.mask_plan else {
            let inner = stream::once(async move {
                let array = decode_segment(
                    segment_source,
                    segment_id,
                    array_tree,
                    layout_dtype,
                    layout_row_count,
                    array_ctx,
                    &session,
                )
                .await?;
                let array = slice_to_range(array, &row_range_for_slice)?;
                array.apply(&expr)
            }
            .map(|res: VortexResult<ArrayRef>| res));
            return Ok(Box::pin(ArrayStreamAdapter::new(dtype, inner)));
        };

        // Slow path: execute the mask plan, fold its bool stream into
        // a single `Mask`, then decode + slice + filter + apply.
        let mask_stream = mask_plan.execute(row_range, ctx)?;
        let stream = try_stream! {
            // Lockstep contract: await enough mask rows to cover this
            // flat layout's row range, then issue the read. (Today
            // the mask plan emits a stream that we collapse into one
            // `Mask`; the partial-read variant lands later — see
            // `LAYOUT_PLAN.md` § FilterPlan and its pushdown.)
            let mask_array = mask_stream.read_all().await?;
            let mut ctx_exec = session.create_execution_ctx();
            let mask: Mask = mask_array.execute::<Mask>(&mut ctx_exec)?;

            let array = decode_segment(
                segment_source,
                segment_id,
                array_tree,
                layout_dtype,
                layout_row_count,
                array_ctx,
                &session,
            )
            .await?;
            let array = slice_to_range(array, &row_range_for_slice)?;
            // Filter first (consistent with the V1 evaluator); the
            // expression sees the mask-filtered rows.
            let array = if mask.all_true() {
                array
            } else {
                array.filter(mask)?
            };
            yield array.apply(&expr)?;
        };
        Ok(Box::pin(ArrayStreamAdapter::new(dtype, stream)))
    }
}

/// Fetch the segment, parse the array tree (either from layout
/// metadata or from the segment itself), and decode into an
/// `ArrayRef` of length `layout_row_count`.
async fn decode_segment(
    segment_source: Arc<dyn SegmentSource>,
    segment_id: SegmentId,
    array_tree: Option<ByteBuffer>,
    dtype: DType,
    layout_row_count: u64,
    array_ctx: ReadContext,
    session: &vortex_session::VortexSession,
) -> VortexResult<ArrayRef> {
    let row_count_usize = usize::try_from(layout_row_count)
        .map_err(|_| vortex_err!("FlatPlan: layout row count exceeds usize"))?;
    let segment_fut = segment_source.request(segment_id);
    let segment = segment_fut.await?;
    let parts = if let Some(tree) = array_tree {
        SerializedArray::from_flatbuffer_and_segment(tree, segment)?
    } else {
        SerializedArray::try_from(segment)?
    };
    parts.decode(&dtype, row_count_usize, &array_ctx, session)
}

/// Slice `array` to `row_range` if it doesn't already cover the full
/// length. Cheap when called with `0..len`.
fn slice_to_range(array: ArrayRef, row_range: &Range<u64>) -> VortexResult<ArrayRef> {
    let start = usize::try_from(row_range.start)
        .map_err(|_| vortex_err!("FlatPlan: row range start exceeds usize"))?;
    let end = usize::try_from(row_range.end)
        .map_err(|_| vortex_err!("FlatPlan: row range end exceeds usize"))?;
    if start == 0 && end == array.len() {
        return Ok(array);
    }
    array.slice(start..end)
}
