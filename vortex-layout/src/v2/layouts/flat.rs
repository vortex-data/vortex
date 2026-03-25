// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;
use vortex_session::registry::ReadContext;

use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::v2::layout::ChildRelationship;
use crate::v2::layout::Layout;
use crate::v2::layout::LayoutChild;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutVTable;
use crate::v2::scan::plan::SegmentRequest;
use crate::v2::scan::planner::NodeId;
use crate::v2::scan::planner::PlanBuilder;
use crate::v2::scan::planner::SplitPlanner;
use crate::v2::scan::planner::SplitPlannerRef;
use crate::v2::selection::Selection;

/// The flat layout vtable.
#[derive(Clone)]
pub struct Flat;

/// Metadata for a flat layout (no additional metadata needed).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FlatMetadata {
    array_ctx: ReadContext,
}

impl fmt::Display for FlatMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FlatMetadata")
    }
}

impl LayoutVTable for Flat {
    type Metadata = FlatMetadata;

    fn id(&self) -> LayoutId {
        LayoutId::new_ref("vortex.flat")
    }

    fn deserialize_metadata(
        _metadata: &[u8],
        _dtype: &DType,
        _row_count: u64,
        _children: &[LayoutChild],
        array_ctx: &ReadContext,
    ) -> VortexResult<Self::Metadata> {
        Ok(FlatMetadata {
            array_ctx: array_ctx.clone(),
        })
    }

    fn child_dtype(_layout: &Layout<Self>, _child_idx: usize) -> &DType {
        unreachable!("Flat layout has no children")
    }

    fn child_relationship(_layout: &Layout<Self>, _child_idx: usize) -> ChildRelationship {
        unreachable!("Flat layout has no children")
    }

    fn prepare(
        layout: &Layout<Self>,
        expr: &Expression,
        // TODO(ngates): we probably want to pass this down? Although it should be available
        //  through the "latest" view of the SplitSelection.
        _selection: &Selection,
        row_offset: Option<u64>,
        row_splits: &mut BTreeSet<u64>,
        _session: &VortexSession,
    ) -> VortexResult<SplitPlannerRef> {
        if let Some(offset) = row_offset {
            row_splits.insert(offset);
            row_splits.insert(offset + layout.row_count());
        }

        let segment_source = layout.segment_source().clone();
        let segment_id = layout
            .segments()
            .first()
            .copied()
            .ok_or_else(|| vortex_err!("FlatLayout missing segment"))?;

        Ok(Arc::new(FlatLayoutPlanner {
            dtype: layout.dtype().clone(),
            len: usize::try_from(layout.row_count())
                .map_err(|_| vortex_err!("Layout larger than usize"))?,
            segment_source,
            segment_id,
            expression: expr.clone(),
            array_ctx: layout.metadata().array_ctx.clone(),
        }))
    }
}

struct FlatLayoutPlanner {
    dtype: DType,
    len: usize,
    array_ctx: ReadContext,
    expression: Expression,
    segment_source: Arc<dyn SegmentSource>,
    segment_id: SegmentId,
}

impl SplitPlanner for FlatLayoutPlanner {
    fn plan_split(
        &self,
        row_range: &Range<u64>,
        selection: NodeId,
        builder: &mut PlanBuilder,
    ) -> VortexResult<NodeId> {
        let split_start = usize::try_from(row_range.start)
            .map_err(|_| vortex_err!("row_range start exceeds usize"))?;
        let split_end = usize::try_from(row_range.end)
            .map_err(|_| vortex_err!("row_range end exceeds usize"))?;

        let segment_req = SegmentRequest {
            source: self.segment_source.clone(),
            segment_id: self.segment_id,
        };

        let decoded = builder.decode_segment(
            segment_req,
            self.dtype.clone(),
            self.len,
            self.array_ctx.clone(),
            row_range,
        )?;

        let sliced = if split_start == 0 && split_end == self.len {
            decoded
        } else {
            builder.slice_node(decoded, split_start..split_end, row_range)?
        };

        let filtered = builder.filter_node(sliced, selection, row_range)?;
        let applied = builder.apply_node(filtered, self.expression.clone(), row_range)?;
        builder.optimize_node(applied, row_range)
    }
}
