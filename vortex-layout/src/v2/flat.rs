// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`FlatPlan`] — terminal plan node over a single segment.
//!
//! See `LAYOUT_PLAN.md` § Per-layout `plan` walkthrough / `FlatLayout::plan`.

use std::ops::Range;
use std::sync::Arc;

use futures::FutureExt;
use futures::stream;
use vortex_array::MaskFuture;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::stream::ArrayStreamAdapter;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_scan::selection::Selection;

use crate::LayoutReaderRef;
use crate::v2::plan::LayoutPlan;
use crate::v2::plan::LayoutPlanRef;
use crate::v2::plan::PartitionStats;
use crate::v2::scan_ctx::ScanCtx;

/// Terminal node. Reads one segment, evaluates `expr` against the
/// resulting array, emits a single-partition stream.
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
        }
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

    fn execute(&self, row_range: Range<u64>, _ctx: &ScanCtx) -> VortexResult<SendableArrayStream> {
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
        Ok(Box::pin(ArrayStreamAdapter::new(dtype, inner)))
    }
}
