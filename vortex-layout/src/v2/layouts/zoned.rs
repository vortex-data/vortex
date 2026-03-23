// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::expr::Expression;
use vortex_error::VortexResult;

use crate::v2::layout::ChildRelationship;
use crate::v2::layout::Layout;
use crate::v2::layout::LayoutId;
use crate::v2::layout::LayoutRef;
use crate::v2::layout::LayoutVTable;
use crate::v2::layout::RowSelection;
use crate::v2::layout::SplitIterator;
use crate::v2::plan::PlanBuilder;

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

    fn plan(
        layout: &Layout<Self>,
        expr: &Expression,
        selection: &RowSelection,
        builder: &PlanBuilder,
    ) -> VortexResult<SplitIterator> {
        todo!()
    }
}

impl Layout<Zoned> {
    pub fn zone_map_child(&self) -> LayoutRef {
        self
    }
}
