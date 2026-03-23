// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::BTreeSet;
use std::ops::Range;

use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;

use crate::v2::layout::ChildRelationship;
use crate::v2::layout::Layout;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutRef;
use crate::v2::layout::LayoutVTable;
use crate::v2::layout::RowSelection;
use crate::v2::planner::NodeId;
use crate::v2::planner::PlanBuilder;
use crate::v2::planner::SplitPlanner;
use crate::v2::planner::SplitPlannerRef;
use crate::v2::planner::SplitSelection;

pub struct Zoned;

impl LayoutVTable for Zoned {
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
        todo!()
    }
}

impl Layout<Zoned> {
    /// Returns the data child of the zoned layout.
    pub fn data_child(&self) -> VortexResult<LayoutRef> {
        self.child(0)
    }

    /// Returns the zone map child of the zoned layout.
    pub fn zone_map_child(&self) -> VortexResult<LayoutRef> {
        self.child(1)
    }
}

struct ZonedLayoutPlanner {}

impl SplitPlanner for ZonedLayoutPlanner {
    fn plan_split(
        &self,
        row_range: Range<u64>,
        selection: &SplitSelection,
        builder: &mut PlanBuilder,
    ) -> VortexResult<NodeId> {
        todo!()
    }
}
