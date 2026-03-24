// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::fmt;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::v2::layout::ChildRelationship;
use crate::v2::layout::Layout;
use crate::v2::layout::LayoutChild;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutVTable;
use crate::v2::scan::plan::SegmentRequest;
use crate::v2::scan::planner::NodeId;
use crate::v2::scan::planner::NodeOpts;
use crate::v2::scan::planner::PlanBuilder;
use crate::v2::scan::planner::SplitPlanner;
use crate::v2::scan::planner::SplitPlannerRef;
use crate::v2::selection::Selection;

/// The flat layout vtable.
#[derive(Clone)]
pub struct Flat;

/// Metadata for a flat layout (no additional metadata needed).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct FlatMetadata;

impl fmt::Display for FlatMetadata {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "FlatMetadata")
    }
}

impl LayoutVTable for Flat {
    type Metadata = FlatMetadata;
    type Plan = ();

    fn id(&self) -> LayoutId {
        LayoutId::new_ref("vortex.flat")
    }

    fn deserialize_metadata(
        _metadata: &[u8],
        _dtype: &DType,
        _row_count: u64,
        _children: &[LayoutChild],
    ) -> VortexResult<FlatMetadata> {
        Ok(FlatMetadata)
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
        row_splits: &mut BTreeSet<u64>,
    ) -> VortexResult<SplitPlannerRef> {
        // TODO(ngates): surely we only need one of them
        row_splits.insert(0);
        row_splits.insert(layout.row_count());

        let segment_source = layout.segment_source().clone();
        let segment_id = layout
            .segments()
            .first()
            .copied()
            .ok_or_else(|| vortex_err!("FlatLayout missing segment"))?;

        Ok(Arc::new(FlatLayoutPlanner {
            segment_source,
            segment_id,
            expression: expr.clone(),
        }))
    }
}

struct FlatLayoutPlanner {
    segment_source: Arc<dyn SegmentSource>,
    segment_id: SegmentId,
    expression: Expression,
}

impl SplitPlanner for FlatLayoutPlanner {
    fn plan_split(
        &self,
        row_range: &Range<u64>,
        _selection: NodeId,
        builder: &mut PlanBuilder,
    ) -> VortexResult<NodeId> {
        let expression = self.expression.clone();
        builder.create_node(NodeOpts {
            inputs: &[],
            segments: vec![SegmentRequest {
                source: self.segment_source.clone(),
                segment_id: self.segment_id,
            }],
            lifetime: builder.row_range_lifetime(row_range.clone()),
            compute: move |mut segments: Vec<ByteBuffer>, _inputs: Vec<ArrayRef>| {
                // The segment is deserialized into an array by the scheduler.
                let _buffer = segments.remove(0);
                // TODO: deserialize buffer into array and evaluate expression.
                todo!("deserialize segment buffer and apply expression")
            },
        })
    }
}
