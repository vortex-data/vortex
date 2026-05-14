// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`FlatPlan`] — terminal plan node over a single segment.
//!
//! See `LAYOUT_PLAN.md` § Per-layout `plan` walkthrough / `FlatLayout::plan`.

use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_array::stream::SendableArrayStream;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_scan::selection::Selection;
use vortex_session::VortexSession;

use crate::LayoutReaderRef;
use crate::v2::plan::{LayoutPlan, LayoutPlanRef, PartitionStats};

/// Terminal node. Reads one segment, evaluates `expr` against the
/// resulting array, emits a single-partition stream.
///
/// In PR 2 this is a scaffold — `execute` is unimplemented. PR 3
/// fills it in by driving the underlying [`crate::LayoutReader`]
/// (and later, sub-segment reads).
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
        Ok(PartitionStats::unknown().with_row_count(self.row_count))
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

    fn execute(
        &self,
        _partition: usize,
        _session: &VortexSession,
    ) -> VortexResult<SendableArrayStream> {
        todo!("FlatPlan::execute — implemented in PR 3 alongside the differential test")
    }
}
