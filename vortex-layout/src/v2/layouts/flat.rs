// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;

use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;

use crate::v2::layout::ChildRelationship;
use crate::v2::layout::Layout;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutVTable;
use crate::v2::layout::RowSelection;
use crate::v2::layout::SplitIterator;
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
        selection: &RowSelection,
        row_splits: &mut BTreeSet<u64>,
        builder: &mut PlanBuilder,
    ) -> VortexResult<SplitPlannerRef> {
        let segment_source = layout.segment_source().clone();
        let segment_id = layout
            .segments()
            .get(0)
            .copied()
            .ok_or_else(|| vortex_err!("FlatLayout missing segment"))?;

        // FIXME(ngates): do we need to know our "coordinates" within the overall scan? That would
        //  help us construct an accurate lifetime.
        // let lifetime = Lifetime::RowRange();

        // TODO(ngates): to do sub-segment reads, we wrap everything in a deferred node until we
        //  get a mask, then we construct the array with placeholder "LazyBufferHandle", run the
        //  optimizer, then extract the byte ranges from the lazy buffer handles to submit the
        //  read.

        todo!()
    }
}

struct FlatLayoutPlanner {}

impl SplitPlanner for FlatLayoutPlanner {
    fn plan_split(
        &self,
        row_range: Range<u64>,
        selection: &SplitSelection,
        builder: &mut PlanBuilder,
    ) -> VortexResult<NodeId> {
        todo!()
    }
}
