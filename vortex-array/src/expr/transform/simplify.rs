// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::expr::exprs::get_item::GetItem;
use crate::expr::exprs::pack::Pack;
use crate::expr::transform::match_between::find_between;
use crate::expr::transform::traits::{ChildReduceRule, RewriteContext};
use crate::expr::traversal::{NodeExt, Transformed};
use crate::expr::{Expression, ExpressionView};

/// Simplifies an expression into an equivalent expression which is faster and easier to analyze.
///
/// If the scope dtype is known, see `simplify_typed` for a simplifier which uses dtype.
pub fn simplify(e: Expression) -> VortexResult<Expression> {
    // Apply pack/get_item simplification directly (no dtype context needed)
    let e = e
        .transform_up(|node| {
            // pack(l_1: e_1, ..., l_i: e_i, ..., l_n: e_n).get_item(l_i) = e_i where 0 <= i <= n
            if let Some(get_item) = node.as_opt::<GetItem>()
                && let Some(pack) = get_item.child(0).as_opt::<Pack>()
            {
                let expr = pack.field(get_item.data())?;
                return Ok(Transformed::yes(expr));
            }
            Ok(Transformed::no(node))
        })
        .map(|e| e.into_inner())?;

    Ok(find_between(e))
}

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

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, PType};

    use super::{PackGetItemRule, simplify};
    use crate::expr::exprs::get_item::{GetItem, get_item};
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::pack::pack;
    use crate::expr::transform::traits::{ChildReduceRule, SimpleRewriteContext};

    #[test]
    fn test_simplify() {
        let e = get_item("b", pack([("a", lit(1)), ("b", lit(2))], NonNullable));
        let e = simplify(e).unwrap();
        assert_eq!(&e, &lit(2));
    }

    #[test]
    fn test_pack_get_item_rule() {
        let rule = PackGetItemRule;

        // Create: pack(a: lit(1), b: lit(2)).get_item("b")
        let pack_expr = pack([("a", lit(1)), ("b", lit(2))], NonNullable);
        let get_item_expr = get_item("b", pack_expr.clone());

        // Create a dummy context
        let dtype = DType::Primitive(PType::I32, NonNullable);
        let ctx = SimpleRewriteContext { dtype: &dtype };

        // Apply the rule - need to downcast to GetItem view
        let get_item_view = get_item_expr.as_::<GetItem>();
        let result = rule
            .reduce_child(&get_item_view, &pack_expr, 0, &ctx)
            .unwrap();

        // Should return Some(lit(2))
        assert!(result.is_some());
        assert_eq!(&result.unwrap(), &lit(2));
    }

    #[test]
    fn test_pack_get_item_rule_no_match() {
        let rule = PackGetItemRule;

        // Create: get_item("x", lit(42)) - not a pack child
        let lit_expr = lit(42);
        let get_item_expr = get_item("x", lit_expr.clone());

        let dtype = DType::Primitive(PType::I32, NonNullable);
        let ctx = SimpleRewriteContext { dtype: &dtype };

        // Apply the rule - need to downcast to GetItem view
        let get_item_view = get_item_expr.as_::<GetItem>();
        let result = rule
            .reduce_child(&get_item_view, &lit_expr, 0, &ctx)
            .unwrap();

        // Should return None (no match)
        assert!(result.is_none());
    }

    #[test]
    fn test_pack_get_item_rule_from_session() {
        use crate::expr::ExprId;
        use crate::expr::session::ExprSession;

        // Create a default session
        let session = ExprSession::default();

        // Get the rewrite rules for GetItem
        let get_item_id = ExprId::new_ref("vortex.get_item");
        let rules = session.rewrite_rules().child_rules_for(&get_item_id);

        // Should have at least one rule registered (PackGetItemRule)
        assert!(rules.is_some());
        assert_eq!(rules.unwrap().len(), 1);

        // Verify the rule works
        let pack_expr = pack([("a", lit(1)), ("b", lit(2))], NonNullable);
        let get_item_expr = get_item("b", pack_expr.clone());

        let dtype = DType::Primitive(PType::I32, NonNullable);
        let ctx = SimpleRewriteContext { dtype: &dtype };

        let rule = &rules.unwrap()[0];
        let result = rule
            .reduce_child_dyn(&get_item_expr, &pack_expr, 0, &ctx)
            .unwrap();

        assert!(result.is_some());
        assert_eq!(&result.unwrap(), &lit(2));
    }

    #[test]
    fn test_multi_level_pack_get_item_simplify() {
        use crate::expr::session::ExprSession;
        use crate::expr::transform::simplify_typed::apply_child_rules;

        // Create a default session
        let session = ExprSession::default();

        // Build nested expression: get_item("z", pack(x: get_item("a", pack(a: lit(1), b: lit(2))), y: lit(3), z: lit(4)))
        // Inner pack
        let inner_pack = pack([("a", lit(1)), ("b", lit(2))], NonNullable);
        let get_a = get_item("a", inner_pack);

        // Outer pack
        let outer_pack = pack([("x", get_a), ("y", lit(3)), ("z", lit(4))], NonNullable);
        let get_z = get_item("z", outer_pack);

        let dtype = DType::Primitive(PType::I32, NonNullable);

        // Apply child rules (bottom-up)
        let result = apply_child_rules(get_z, &dtype, &session).unwrap();

        // Should simplify all the way down to lit(4)
        // Bottom-up: first inner get_item("a", pack(...)) -> lit(1)
        // Then outer get_item("z", pack(x: lit(1), y: lit(3), z: lit(4))) -> lit(4)
        assert_eq!(&result, &lit(4));
    }

    #[test]
    fn test_deeply_nested_pack_get_item() {
        use crate::expr::session::ExprSession;
        use crate::expr::transform::simplify_typed::apply_child_rules;

        let session = ExprSession::default();

        // Build: get_item("final", pack(final: get_item("c", pack(c: get_item("b", pack(b: get_item("a", pack(a: lit(42)))))))))
        let innermost = pack([("a", lit(42))], NonNullable);
        let get_a = get_item("a", innermost);

        let level2 = pack([("b", get_a)], NonNullable);
        let get_b = get_item("b", level2);

        let level3 = pack([("c", get_b)], NonNullable);
        let get_c = get_item("c", level3);

        let outermost = pack([("final", get_c)], NonNullable);
        let get_final = get_item("final", outermost);

        let dtype = DType::Primitive(PType::I32, NonNullable);

        // Apply bottom-up simplification
        let result = apply_child_rules(get_final, &dtype, &session).unwrap();

        // Should collapse all the way to lit(42)
        assert_eq!(&result, &lit(42));
    }

    #[test]
    fn test_partial_pack_get_item_simplify() {
        use crate::expr::exprs::binary::checked_add;
        use crate::expr::session::ExprSession;
        use crate::expr::transform::simplify_typed::apply_child_rules;

        let session = ExprSession::default();

        // Build: get_item("result", pack(result: add(get_item("x", pack(x: lit(1), y: lit(2))), lit(10))))
        // The inner get_item should simplify, but outer structure remains
        let inner_pack = pack([("x", lit(1)), ("y", lit(2))], NonNullable);
        let get_x = get_item("x", inner_pack);
        let add_expr = checked_add(get_x, lit(10));

        let outer_pack = pack([("result", add_expr)], NonNullable);
        let get_result = get_item("result", outer_pack);

        let dtype = DType::Primitive(PType::I32, NonNullable);

        // Apply bottom-up simplification
        let result = apply_child_rules(get_result, &dtype, &session).unwrap();

        // Should simplify to: add(lit(1), lit(10))
        let expected = checked_add(lit(1), lit(10));
        assert_eq!(&result, &expected);
    }

    #[test]
    fn test_simplify_with_rules_api() {
        use crate::expr::session::ExprSession;
        use crate::expr::transform::simplify_typed::simplify_with_session;

        let session = ExprSession::default();

        // Build a multi-level expression
        let inner = pack([("a", lit(100)), ("b", lit(200))], NonNullable);
        let middle = pack([("x", get_item("a", inner)), ("y", lit(300))], NonNullable);
        let outer = get_item("x", middle);

        let dtype = DType::Primitive(PType::I32, NonNullable);

        // Use the convenience API
        let result = simplify_with_session(outer, &dtype, &session).unwrap();

        // Should fully simplify to lit(100)
        assert_eq!(&result, &lit(100));
    }
}
