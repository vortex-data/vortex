// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;
use std::sync::Arc;

use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::segments::SegmentId;
use crate::segments::SegmentSource;
use crate::v2::layout::ChildRelationship;
use crate::v2::layout::Layout;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutVTable;
use crate::v2::layout::RowSelection;
use crate::v2::plan::Lifetime;
use crate::v2::planner::NodeId;
use crate::v2::planner::NodeOpts;
use crate::v2::planner::PlanBuilder;
use crate::v2::planner::SplitPlanner;
use crate::v2::planner::SplitPlannerRef;
use crate::v2::planner::SplitSelection;

pub struct Flat;

impl LayoutVTable for Flat {
    type Metadata = ();
    type Plan = ();

    fn id(&self) -> LayoutId {
        todo!()
    }

    fn child_dtype(layout: &Layout<Self>, child_idx: usize) -> &DType {
        todo!()
    }

    fn child_relationship(layout: &Layout<Self>, child_idx: usize) -> ChildRelationship {
        todo!()
    }

    fn prepare(
        layout: &Layout<Self>,
        expr: &Expression,
        // TODO(ngates): we probably want to pass this down? Although it should be available
        //  through the "latest" view of the SplitSelection.
        _selection: &RowSelection,
        row_splits: &mut BTreeSet<u64>,
        _builder: &mut PlanBuilder,
    ) -> VortexResult<SplitPlannerRef> {
        // TODO(ngates): surely we only need one of them
        row_splits.insert(0);
        row_splits.insert(layout.row_count());

        let segment_source = layout.segment_source().clone();
        let segment_id = layout
            .segments()
            .get(0)
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
        row_range: Range<u64>,
        selection: &SplitSelection,
        builder: &mut PlanBuilder,
    ) -> VortexResult<NodeId> {
        builder.create_node(&NodeOpts {
            inputs: &[selection.node_id()],
            segments: &[],
            lifetime: Lifetime::Scan,
            compute: |inputs| {
                let buffer = inputs[0].into_buffer();
                todo!()
            },
        })
    }
}
