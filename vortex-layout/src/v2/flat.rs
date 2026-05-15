// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`FlatPlan`] — terminal plan node over a single segment.
//!
//! Optionally absorbs a `mask_plan` (via [`LayoutPlan::try_pushdown_mask`])
//! and forwards its result into the underlying reader's
//! `projection_evaluation` mask argument. Saves a separate
//! `FilterPlan` wrap and lets the reader exploit the mask (today: CPU
//! savings on filter eval; eventually: sub-segment reads via
//! `FuseFilterIntoFlat`).
//!
//! See `LAYOUT_PLAN.md` § Per-layout `plan` walkthrough / `FlatLayout::plan`
//! and § FilterPlan and its pushdown.

use std::ops::Range;
use std::sync::Arc;

use async_stream::try_stream;
use futures::FutureExt;
use futures::stream;
use vortex_array::MaskFuture;
use vortex_array::VortexSessionExecute;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::ArrayStreamExt;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_mask::Mask;
use vortex_scan::selection::Selection;

use crate::LayoutReaderRef;
use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;
use crate::v2::scan_ctx::ScanCtx;

/// Terminal node. Reads one segment, evaluates `expr` against the
/// resulting array, emits a single-partition stream.
///
/// `mask_plan`, when set, is executed first and its bool result is
/// passed to the underlying reader's `projection_evaluation` mask —
/// see [`Self::with_mask`]. Without a mask the reader sees an
/// always-true mask of the requested length.
///
/// Today this bridges to the v1 [`crate::LayoutReader`] under the hood.
/// The v2 trait exposes the right shape for future sub-segment reads
/// and `FilterPlan` fusion (see `LAYOUT_PLAN.md` § FilterPlan).
pub struct FlatPlan {
    reader: LayoutReaderRef,
    expr: Expression,
    selection: Selection,
    output_dtype: DType,
    row_count: u64,
    /// Optional pushed-down mask. If set, executed at execute-time
    /// over the same row range and forwarded to the reader. Schema
    /// must be `DType::Bool(_)` and the row space must match this
    /// plan's.
    mask_plan: Option<LayoutPlanRef>,
}

impl FlatPlan {
    pub fn new(
        reader: LayoutReaderRef,
        expr: Expression,
        selection: Selection,
        output_dtype: DType,
        row_count: u64,
    ) -> Self {
        Self {
            reader,
            expr,
            selection,
            output_dtype,
            row_count,
            mask_plan: None,
        }
    }

    /// Return a copy of this plan with `mask_plan` pushed in. Caller
    /// (`FilterPlan` via [`LayoutPlan::try_pushdown_mask`]) should
    /// drop its outer `FilterPlan` wrapper after taking the rewrite.
    pub fn with_mask(mut self, mask_plan: LayoutPlanRef) -> Self {
        debug_assert!(
            matches!(mask_plan.schema(), DType::Bool(_)),
            "FlatPlan::with_mask: mask_plan must produce a Bool stream"
        );
        self.mask_plan = Some(mask_plan);
        self
    }

    pub fn reader(&self) -> &LayoutReaderRef {
        &self.reader
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
        Ok(PartitionStats::for_range(0..self.row_count))
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
        // Stack masks: if there's already one, AND the new in via a
        // tiny wrapper plan. For the only caller today (FilterPlan),
        // there's never an existing mask, but defining the behaviour
        // makes future composition explicit.
        if self.mask_plan.is_some() {
            return None; // Would need an AndBool layer; skip for now.
        }
        Some(Arc::new(Self {
            reader: Arc::clone(&self.reader),
            expr: self.expr.clone(),
            selection: self.selection.clone(),
            output_dtype: self.output_dtype.clone(),
            row_count: self.row_count,
            mask_plan: Some(mask_plan),
        }))
    }

    fn execute(&self, row_range: Range<u64>, ctx: &ScanCtx) -> VortexResult<SendableArrayStream> {
        if !matches!(self.selection, Selection::All) {
            // The V2 entrypoints never hand FlatPlan a non-`All`
            // selection — `FilterPlan` carries masks separately.
            vortex_bail!("FlatPlan only supports Selection::All in the projection-only path");
        }
        if row_range.start > self.row_count || row_range.end > self.row_count {
            vortex_bail!(
                "FlatPlan::execute row range {row_range:?} exceeds layout row count {}",
                self.row_count
            );
        }

        // Fast path: no pushed-down mask. Read with an all-true mask
        // and a one-shot stream of the resulting array.
        let Some(mask_plan) = &self.mask_plan else {
            let row_count_usize: usize =
                (row_range.end - row_range.start).try_into().map_err(|_| {
                    vortex_error::vortex_err!(
                        "FlatPlan::execute row range too large for usize: {row_range:?}",
                    )
                })?;
            let mask = MaskFuture::new_true(row_count_usize);
            let array_fut = self
                .reader
                .projection_evaluation(&row_range, &self.expr, mask)?;
            let dtype = self.output_dtype.clone();
            let inner = stream::once(array_fut.map(|res| res));
            return Ok(Box::pin(ArrayStreamAdapter::new(dtype, inner)));
        };

        // Slow path: execute the mask plan, fold its bool stream into
        // a single `Mask`, then read with that mask. The reader uses
        // it to filter the output (and, in future, to skip
        // sub-segment reads).
        let mask_stream = mask_plan.execute(row_range.clone(), ctx)?;
        let session = ctx.session().clone();
        let dtype = self.output_dtype.clone();
        let reader = Arc::clone(&self.reader);
        let expr = self.expr.clone();
        let row_range_clone = row_range;

        let stream = try_stream! {
            let mask_array = mask_stream.read_all().await?;
            let mut ctx_exec = session.create_execution_ctx();
            let mask: Mask = mask_array.execute::<Mask>(&mut ctx_exec)?;
            let mask_fut = MaskFuture::ready(mask);
            let array_fut = reader.projection_evaluation(&row_range_clone, &expr, mask_fut)?;
            yield array_fut.await?;
        };
        Ok(Box::pin(ArrayStreamAdapter::new(dtype, stream)))
    }
}
