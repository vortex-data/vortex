// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`FilterPlan`] — applies a per-row mask stream to a value stream.
//!
//! `FilterPlan(value_plan, mask_plan)` is the only node that actually
//! filters in the v2 design. At execute time it consumes value and
//! mask batches in row-aligned lockstep (via [`AlignedArrayStream`])
//! and emits filtered value batches.
//!
//! The PR4 invariant of "no plan-internal caches" still applies. The
//! `RowDemand` publish that the design doc calls for (so source
//! plans can short-circuit on rejected rows) is wired in alongside
//! the real `RowDemand` implementation; for now the producer is a
//! no-op stub.
//!
//! See `LAYOUT_PLAN.md` § FilterPlan and its pushdown.

use std::ops::Range;
use std::sync::Arc;

use futures::StreamExt;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_io::session::RuntimeSessionExt;
use vortex_mask::Mask;

use crate::v2::aligned::AlignedArrayStream;
use crate::v2::demand::RowDemand;
use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;
use crate::v2::scan_ctx::ScanCtx;

/// Applies `mask` to `values` per row. Output dtype matches the
/// value plan's schema; output row count is the number of `true`
/// rows in the mask, not the input row count.
pub struct FilterPlan {
    values: LayoutPlanRef,
    mask: LayoutPlanRef,
    output_dtype: DType,
}

impl FilterPlan {
    /// Construct a `FilterPlan` over `values` and `mask`. Always
    /// returns a real `FilterPlan`; use [`Self::new_or_pushdown`] to
    /// give the values plan a chance to absorb the mask first.
    pub fn new(values: LayoutPlanRef, mask: LayoutPlanRef) -> Self {
        debug_assert!(
            matches!(mask.schema(), DType::Bool(_)),
            "FilterPlan: mask plan must produce a Bool stream",
        );
        let output_dtype = values.schema().clone();
        Self {
            values,
            mask,
            output_dtype,
        }
    }

    /// Try to push `mask` into `values` via
    /// [`LayoutPlan::try_pushdown_mask`]. If the values plan absorbs
    /// it, return the rewrite directly (no `FilterPlan` wrapper). If
    /// not, fall back to wrapping with `FilterPlan::new`.
    pub fn new_or_pushdown(values: LayoutPlanRef, mask: LayoutPlanRef) -> LayoutPlanRef {
        debug_assert!(
            matches!(mask.schema(), DType::Bool(_)),
            "FilterPlan: mask plan must produce a Bool stream",
        );
        if let Some(pushed) = Arc::clone(&values).try_pushdown_mask(Arc::clone(&mask)) {
            return pushed;
        }
        Arc::new(Self::new(values, mask))
    }
}

impl PartialEq for FilterPlan {
    fn eq(&self, other: &Self) -> bool {
        crate::v2::plan::plans_eq(&self.values, &other.values)
            && crate::v2::plan::plans_eq(&self.mask, &other.mask)
            && self.output_dtype == other.output_dtype
    }
}

impl Eq for FilterPlan {}

impl std::hash::Hash for FilterPlan {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        crate::v2::plan::hash_plan(&self.values, state);
        crate::v2::plan::hash_plan(&self.mask, state);
        self.output_dtype.hash(state);
    }
}

impl LayoutPlan for FilterPlan {
    fn schema(&self) -> &DType {
        &self.output_dtype
    }

    fn partition_count(&self) -> usize {
        // Row-range partitioning passes through. We don't merge
        // values/mask partitions because both are derived from the
        // same Layout::plan and share the same partitioning shape.
        self.values.partition_count()
    }

    fn partition_stats(&self, partition: usize) -> VortexResult<PartitionStats> {
        // The row range is in the *input* row coordinate space. The
        // actual emitted row count after filtering is data-dependent
        // and unknown without executing.
        self.values.partition_stats(partition)
    }

    fn output_ordered(&self) -> bool {
        self.values.output_ordered() && self.mask.output_ordered()
    }

    fn required_input_ordered(&self) -> Vec<bool> {
        vec![true, true]
    }

    fn maintains_input_order(&self) -> Vec<bool> {
        vec![true, false]
    }

    fn children(&self) -> &[LayoutPlanRef] {
        // Children order: [values, mask].
        // Returning an empty slice here would be safe (we just won't
        // be visited by the pushdown walker), but exposing the real
        // children lets PR6 pushdown rules find them.
        // We can't return `&[values, mask]` because they're not
        // contiguous in memory — would need an owning vec on each
        // call. Skip for now; PR6 can add a `children_arc` accessor
        // if it needs them.
        &[]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<LayoutPlanRef>,
    ) -> VortexResult<LayoutPlanRef> {
        if !children.is_empty() {
            vortex_bail!("FilterPlan does not yet expose its children for replacement");
        }
        Ok(self)
    }

    fn execute(
        &self,
        row_range: Range<u64>,
        demand: &RowDemand,
        ctx: &ScanCtx,
    ) -> VortexResult<SendableArrayStream> {
        let values_stream = self.values.execute(row_range.clone(), demand, ctx)?;
        let mask_stream = self.mask.execute(row_range, demand, ctx)?;

        let session = ctx.session().clone();
        let dtype = self.output_dtype.clone();
        let aligned =
            AlignedArrayStream::new(vec![values_stream, mask_stream], ctx.session().handle());
        let mapped = aligned.map(move |result| {
            let mut arrays = result?.into_iter();
            let values = arrays
                .next()
                .vortex_expect("FilterPlan: values stream missing from aligned output");
            let mask = arrays
                .next()
                .vortex_expect("FilterPlan: mask stream missing from aligned output");
            // Convert the bool array to a `Mask` and apply. Same
            // round-trip the v1 `FlatReader::projection_evaluation`
            // does when it has a non-trivial mask.
            let mut ctx = session.create_execution_ctx();
            let mask: Mask = mask.execute::<Mask>(&mut ctx)?;
            if mask.all_true() {
                Ok(values)
            } else {
                values.filter(mask)
            }
        });

        Ok(Box::pin(ArrayStreamAdapter::new(dtype, mapped)))
    }
}
