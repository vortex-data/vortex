// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::expr::exprs::get_item::GetItem;
use crate::expr::exprs::pack::Pack;
use crate::expr::transform::rules::{ChildReduceRule, RewriteContext};
use crate::expr::{Expression, ExpressionView};

/// Rewrite rule: `pack(l_1: e_1, ..., l_i: e_i, ..., l_n: e_n).get_item(l_i) = e_i`
///
/// Simplifies accessing a field from a pack expression by directly returning the field's
/// expression instead of materializing the pack.
///
/// # Example
/// ```
/// # use vortex_array::expr::exprs::{get_item::get_item, literal::lit, pack::pack};
/// # use vortex_dtype::Nullability::NonNullable;
/// let e = get_item("b", pack([("a", lit(1)), ("b", lit(2))], NonNullable));
/// // After applying PackGetItemRule, this becomes: lit(2)
/// ```
pub struct PackGetItemRule;

impl ChildReduceRule<GetItem> for PackGetItemRule {
    fn reduce_child(
        &self,
        get_item: &ExpressionView<GetItem>,
        child: &Expression,
        child_idx: usize,
        _ctx: &dyn RewriteContext,
    ) -> VortexResult<Option<Expression>> {
        // Only consider the first child (child_idx == 0) of GetItem expressions
        if child_idx != 0 {
            return Ok(None);
        }

        // Check if child is Pack
        if let Some(pack) = child.as_opt::<Pack>() {
            // Extract the field from the pack
            let field_expr = pack.field(get_item.data())?;
            return Ok(Some(field_expr));
        }

        Ok(None)
    }
}
