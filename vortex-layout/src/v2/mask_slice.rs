// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`MaskSlicePlan`] — adapter that re-bases a mask plan into a
//! sub-range's local row coordinates.
//!
//! Used by [`crate::v2::chunked::ChunkedPlan::try_pushdown_mask`] when
//! it needs to give each chunk's child plan only the chunk's slice of
//! a parent mask. The wrapped child sees a plan in its own
//! coordinate space; at execute time the wrapper translates back to
//! the parent's absolute coordinates and calls the original mask.
//!
//! This is the simple, redundant version: each chunk's mask read is
//! independent. Shared evaluation across chunks waits for the CSE
//! pass + `Let` / `Use` (see `LAYOUT_PLAN.md` § Tee and CSE).

use std::ops::Range;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::v2::dataflow::LayoutLoweringCtx;
use crate::v2::dataflow::OutputFrontier;
use crate::v2::demand::RowDemand;
use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;
use crate::v2::scan_ctx::ScanCtx;

/// Re-bases an inner mask plan into a sub-range's local coordinates.
/// The wrapped plan's row space is `0..slice_len`; `execute` adds
/// `slice_offset` and forwards to the inner plan.
pub struct MaskSlicePlan {
    inner: LayoutPlanRef,
    slice_offset: u64,
    slice_len: u64,
}

impl MaskSlicePlan {
    /// Build an adapter exposing `inner` as a plan over the local
    /// range `0..(abs_range.end - abs_range.start)`. `inner` must
    /// already cover at least `abs_range`.
    pub fn new(inner: LayoutPlanRef, abs_range: Range<u64>) -> Self {
        debug_assert!(abs_range.end >= abs_range.start);
        Self {
            inner,
            slice_offset: abs_range.start,
            slice_len: abs_range.end - abs_range.start,
        }
    }
}

impl PartialEq for MaskSlicePlan {
    fn eq(&self, other: &Self) -> bool {
        crate::v2::plan::plans_eq(&self.inner, &other.inner)
            && self.slice_offset == other.slice_offset
            && self.slice_len == other.slice_len
    }
}

impl Eq for MaskSlicePlan {}

impl std::hash::Hash for MaskSlicePlan {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        crate::v2::plan::hash_plan(&self.inner, state);
        self.slice_offset.hash(state);
        self.slice_len.hash(state);
    }
}

impl LayoutPlan for MaskSlicePlan {
    fn schema(&self) -> &DType {
        self.inner.schema()
    }

    fn partition_count(&self) -> usize {
        1
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        if partition >= 1 {
            vortex_bail!("MaskSlicePlan partition out of range: {partition}");
        }
        Ok(PartitionStats::for_range(0..self.slice_len))
    }

    fn output_ordered(&self) -> bool {
        self.inner.output_ordered()
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        self.inner.required_input_ordered()
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        self.inner.maintains_input_order()
    }

    fn children(&self) -> &[LayoutPlanRef] {
        std::slice::from_ref(&self.inner)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        if children.len() != 1 {
            vortex_bail!(
                "MaskSlicePlan::with_new_children expected 1 child, got {}",
                children.len()
            );
        }
        let inner = children
            .into_iter()
            .next()
            .ok_or_else(|| vortex_error::vortex_err!("MaskSlicePlan: empty children vec"))?;
        Ok(Arc::new(Self {
            inner,
            slice_offset: self.slice_offset,
            slice_len: self.slice_len,
        }))
    }

    fn lower_to_scheduler(
        &self,
        row_range: Range<u64>,
        ctx: &mut LayoutLoweringCtx,
    ) -> VortexResult<()> {
        if row_range.end > self.slice_len {
            vortex_bail!(
                "MaskSlicePlan::lower_to_scheduler row range {row_range:?} exceeds slice len {}",
                self.slice_len
            );
        }

        let global_range = ctx.current_global_range();
        ctx.register_plan_node(row_range.clone(), self.schema(), 1);
        let abs_start = self.slice_offset + row_range.start;
        let abs_end = self.slice_offset + row_range.end;
        ctx.with_global_range(global_range, |ctx| {
            self.inner.lower_to_scheduler(abs_start..abs_end, ctx)
        })
    }

    fn execute(
        &self,
        row_range: Range<u64>,
        _demand: &RowDemand,
        frontier: &OutputFrontier,

        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        if row_range.end > self.slice_len {
            vortex_bail!(
                "MaskSlicePlan::execute row range {row_range:?} exceeds slice len {}",
                self.slice_len
            );
        }
        let abs_start = self.slice_offset + row_range.start;
        let abs_end = self.slice_offset + row_range.end;
        // Inner plan operates in a wider coord system (full mask
        // range) than what we received; pass detached so it doesn't
        // mis-attribute publishes against our parent's narrow scope.
        let inner_demand = RowDemand::empty(abs_end);
        self.inner
            .execute(abs_start..abs_end, &inner_demand, frontier, ctx)
    }
}
