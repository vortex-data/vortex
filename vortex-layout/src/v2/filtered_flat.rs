// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`FilteredFlatPlan`] — terminal plan node over one segment with a
//! pushed-down mask plan.
//!
//! Produced by [`crate::v2::flat::FlatPlan::try_pushdown_mask`]
//! when a `FilterPlan`'s mask is absorbed into the leaf. At execute
//! time it folds the mask stream into a single [`Mask`], decodes the
//! segment, slices to the requested row range, filters, and applies
//! the projection expression.
//!
//! Sub-segment-aware reads (only fetch the bytes the mask still
//! demands) land later (see `FuseFilterIntoFlat` in `LAYOUT_PLAN.md`).
//!
//! See `LAYOUT_PLAN.md` § FilterPlan and its pushdown.

use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;

use async_stream::try_stream;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
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
use crate::v2::flat::decode_segment;
use crate::v2::flat::slice_to_range;
use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;
use crate::v2::scan_ctx::ScanCtx;

/// Terminal node over one segment with a pushed-down mask. The mask
/// plan is the sole child — the CSE pass collapses N
/// `FilteredFlatPlan`s sharing one mask source into a `Let` + `Use`
/// pair so the mask source executes once.
pub struct FilteredFlatPlan {
    segment_id: SegmentId,
    layout_row_count: u64,
    layout_dtype: DType,
    array_ctx: ReadContext,
    array_tree: Option<ByteBuffer>,
    segment_source: Arc<dyn SegmentSource>,
    expr: Expression,
    selection: Selection,
    output_dtype: DType,
    mask_plan: LayoutPlanRef,
}

impl FilteredFlatPlan {
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
        mask_plan: LayoutPlanRef,
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
            mask_plan,
        }
    }

    pub fn segment_id(&self) -> SegmentId {
        self.segment_id
    }

    pub fn mask_plan(&self) -> &LayoutPlanRef {
        &self.mask_plan
    }
}

impl PartialEq for FilteredFlatPlan {
    fn eq(&self, other: &Self) -> bool {
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
            && crate::v2::plan::plans_eq(&self.mask_plan, &other.mask_plan)
    }
}

impl Eq for FilteredFlatPlan {}

impl Hash for FilteredFlatPlan {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.segment_id.hash(state);
        self.layout_row_count.hash(state);
        self.layout_dtype.hash(state);
        self.array_tree.hash(state);
        self.expr.hash(state);
        match self.selection {
            Selection::All => state.write_u8(0),
            _ => state.write_u8(1),
        }
        self.output_dtype.hash(state);
        crate::v2::plan::hash_plan(&self.mask_plan, state);
    }
}

impl LayoutPlan for FilteredFlatPlan {
    fn schema(&self) -> &DType {
        &self.output_dtype
    }

    fn partition_count(&self) -> usize {
        1
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        if partition >= 1 {
            vortex_bail!("FilteredFlatPlan partition out of range: {partition}");
        }
        // Row count is the layout's count (upper bound). Post-filter
        // row count isn't known at plan time.
        Ok(PartitionStats::for_range(0..self.layout_row_count))
    }

    fn output_ordered(&self) -> bool {
        true
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        vec![true]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true]
    }

    fn children(&self) -> &[LayoutPlanRef] {
        std::slice::from_ref(&self.mask_plan)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        if children.len() != 1 {
            vortex_bail!(
                "FilteredFlatPlan::with_new_children expected 1 child (mask), got {}",
                children.len()
            );
        }
        let mask_plan = children
            .into_iter()
            .next()
            .ok_or_else(|| vortex_err!("FilteredFlatPlan with_new_children: empty vec"))?;
        Ok(Arc::new(Self {
            segment_id: self.segment_id,
            layout_row_count: self.layout_row_count,
            layout_dtype: self.layout_dtype.clone(),
            array_ctx: self.array_ctx.clone(),
            array_tree: self.array_tree.clone(),
            segment_source: Arc::clone(&self.segment_source),
            expr: self.expr.clone(),
            selection: self.selection.clone(),
            output_dtype: self.output_dtype.clone(),
            mask_plan,
        }))
    }

    fn try_pushdown_mask(self: Arc<Self>, _mask_plan: LayoutPlanRef) -> Option<LayoutPlanRef> {
        // Already filtered. Stacking would need an AND of the two
        // masks; not implemented today.
        None
    }

    fn execute(&self, row_range: Range<u64>, ctx: &ScanCtx) -> VortexResult<SendableArrayStream> {
        if !matches!(self.selection, Selection::All) {
            vortex_bail!(
                "FilteredFlatPlan only supports Selection::All — non-All carried by FilterPlan separately"
            );
        }
        if row_range.start > self.layout_row_count || row_range.end > self.layout_row_count {
            vortex_bail!(
                "FilteredFlatPlan::execute row range {row_range:?} exceeds layout row count {}",
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

        let mask_stream = self.mask_plan.execute(row_range, ctx)?;
        let stream = try_stream! {
            // Lockstep contract: await enough mask rows to cover this
            // flat layout's row range, then issue the read. The
            // partial-read variant lands later (see
            // `LAYOUT_PLAN.md` § FilterPlan and its pushdown).
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
