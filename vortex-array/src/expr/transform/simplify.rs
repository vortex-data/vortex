// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::expr::Expression;
use crate::expr::session::ExprSession;
use crate::expr::transform::match_between::find_between;
use crate::expr::transform::simplify_typed::apply_child_rules;

/// Simplifies an expression into an equivalent expression which is faster and easier to analyze.
///
/// If the scope dtype is known, see `simplify_typed` for a simplifier which uses dtype.
pub fn simplify(e: Expression) -> VortexResult<Expression> {
    // Use a dummy DType since the PackGetItem rule doesn't need dtype context
    let dummy_dtype = DType::Bool(vortex_dtype::Nullability::NonNullable);
    let session = ExprSession::default();
    let e = apply_child_rules(e, &dummy_dtype, &session)?;
    Ok(find_between(e))
}

#[cfg(test)]
mod tests {
    use vortex_dtype::Nullability::NonNullable;
    use vortex_dtype::{DType, PType};

    use crate::expr::ExprId;
    use crate::expr::exprs::binary::checked_add;
    use crate::expr::exprs::get_item::transform::PackGetItemRule;
    use crate::expr::exprs::get_item::{GetItem, get_item};
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::pack::pack;
    use crate::expr::session::ExprSession;
    use crate::expr::transform::simplify_typed::apply_child_rules;
    use crate::expr::transform::traits::{ChildReduceRule, SimpleRewriteContext};

    #[test]
    fn test_pack_get_item_rule() {
        let rule = PackGetItemRule;

        // Create: pack(a: lit(1), b: lit(2)).get_item("b")
        let pack_expr = pack([("a", lit(1)), ("b", lit(2))], NonNullable);
        let get_item_expr = get_item("b", pack_expr.clone());

        // Create a dummy context
        let dtype = DType::Primitive(PType::I32, NonNullable);
        let ctx = SimpleRewriteContext { dtype: &dtype };

        let get_item_view = get_item_expr.as_::<GetItem>();
        let result = rule
            .reduce_child(&get_item_view, &pack_expr, 0, &ctx)
            .unwrap();

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

        let get_item_view = get_item_expr.as_::<GetItem>();
        let result = rule
            .reduce_child(&get_item_view, &lit_expr, 0, &ctx)
            .unwrap();

        assert!(result.is_none());
    }

    #[test]
    fn test_pack_get_item_rule_from_session() {
        let session = ExprSession::default();

        let get_item_id = ExprId::new_ref("vortex.get_item");
        let rules = session.rewrite_rules().child_rules_for(&get_item_id);

        // Should have at least one rule registered (PackGetItemRule)
        assert!(rules.is_some());
        assert_eq!(rules.unwrap().len(), 1);

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
        let session = ExprSession::default();

        let inner_pack = pack([("a", lit(1)), ("b", lit(2))], NonNullable);
        let get_a = get_item("a", inner_pack);

        let outer_pack = pack([("x", get_a), ("y", lit(3)), ("z", lit(4))], NonNullable);
        let get_z = get_item("z", outer_pack);

        let dtype = DType::Primitive(PType::I32, NonNullable);

        let result = apply_child_rules(get_z, &dtype, &session).unwrap();

        assert_eq!(&result, &lit(4));
    }

    #[test]
    fn test_deeply_nested_pack_get_item() {
        let session = ExprSession::default();

        let innermost = pack([("a", lit(42))], NonNullable);
        let get_a = get_item("a", innermost);

        let level2 = pack([("b", get_a)], NonNullable);
        let get_b = get_item("b", level2);

        let level3 = pack([("c", get_b)], NonNullable);
        let get_c = get_item("c", level3);

        let outermost = pack([("final", get_c)], NonNullable);
        let get_final = get_item("final", outermost);

        let dtype = DType::Primitive(PType::I32, NonNullable);

        let result = apply_child_rules(get_final, &dtype, &session).unwrap();

        assert_eq!(&result, &lit(42));
    }

    #[test]
    fn test_partial_pack_get_item_simplify() {
        let session = ExprSession::default();

        let inner_pack = pack([("x", lit(1)), ("y", lit(2))], NonNullable);
        let get_x = get_item("x", inner_pack);
        let add_expr = checked_add(get_x, lit(10));

        let outer_pack = pack([("result", add_expr)], NonNullable);
        let get_result = get_item("result", outer_pack);

        let dtype = DType::Primitive(PType::I32, NonNullable);

        let result = apply_child_rules(get_result, &dtype, &session).unwrap();

        let expected = checked_add(lit(1), lit(10));
        assert_eq!(&result, &expected);
    }
}
