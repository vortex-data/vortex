// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::v2::layout::LayoutRef;
use crate::v2::layouts::expr::ExprLayout;
use crate::v2::optimizer::ReduceParent;
use crate::v2::view::LayoutView;
use crate::v2::vtable::{ChildName, VTable};
use crate::LayoutId;
use vortex_error::VortexResult;

/// A layout that combines one child layout per field into an aligned stream of struct arrays.
pub struct ZonedLayout;

impl VTable for ZonedLayout {
    type Instance = ();

    fn id(&self) -> LayoutId {
        LayoutId::from("vortex.zoned")
    }

    fn child_name(&self, _view: &LayoutView<Self>, child_idx: usize) -> ChildName {
        match child_idx {
            0 => ChildName::from("zone_map"),
            1 => ChildName::from("data"),
            _ => unreachable!(),
        }
    }
}

/// Optimizer rule to replace falsification expressions over zoned layouts with zone stats
/// expressions.
impl ReduceParent<ExprLayout, 0> for ZonedLayout {
    fn reduce_parent(
        layout: &LayoutView<Self>,
        parent: &LayoutView<ExprLayout>,
    ) -> VortexResult<Option<LayoutRef>> {
        // So if the parent is an ExprLayout, we look at its expression.

        // If the expression contains falsifications, then we know we're using the zoned layout
        // rather than the data layout.
        let zoned = layout.child(0)?;

        // for falsify_expr in parent.expr().iter_match(|e| e.is_falsify()) {
        //   if self.present_stats.contains(&falsify_expr.lhs()) {
        //      // We replace the falsify expression with a new expression that uses the zone map's min/max
        //      return RepeatLayout(
        //        child = ExprNode.new(
        //          expr = getitem($, "min"),,
        //          child = zoned,
        //        ),
        //        repeat_each_row = self.zone_length,
        //        repeat_last_row = self.row_count % self.zone_length,
        //      )
        //}

        // in the zone map that we can swap it for.
        // that we can swap it for

        todo!()
    }
}
