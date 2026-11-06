// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::expr::Expression;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_error::VortexExpect;

/// Replaces all occurrences of `needle` in the expression `expr` with `replacement`.
pub fn replace(expr: Expression, needle: &Expression, replacement: Expression) -> Expression {
    expr.transform_down(|node| {
        if &node == needle {
            Ok(Transformed {
                value: replacement.clone(),
                // If there is a match with a needle there can be no more matches in that subtree.
                order: TraversalOrder::Skip,
                changed: true,
            })
        } else {
            Ok(Transformed::no(node))
        }
    })
    .vortex_expect("ReplaceVisitor should not fail")
    .into_inner()
}

#[cfg(test)]
mod test {
    use vortex_array::dtype::Nullability::NonNullable;
    use vortex_array::expr::Lambda;
    use vortex_array::expr::checked_add;
    use vortex_array::expr::col;
    use vortex_array::expr::get_item;
    use vortex_array::expr::list_transform;
    use vortex_array::expr::lit;
    use vortex_array::expr::pack;
    use vortex_array::expr::root;
    use vortex_array::expr::var;
    use vortex_error::VortexResult;

    use super::replace;

    #[test]
    fn test_replace_full_tree() {
        let e = get_item("b", pack([("a", lit(1)), ("b", lit(2))], NonNullable));
        let needle = get_item("b", pack([("a", lit(1)), ("b", lit(2))], NonNullable));
        let replacement = lit(42);
        let replaced_expr = replace(e, &needle, replacement.clone());
        assert_eq!(&replaced_expr, &replacement);
    }

    #[test]
    fn test_replace_leaf() {
        let e = pack([("a", lit(1)), ("b", lit(2))], NonNullable);
        let needle = lit(2);
        let replacement = lit(42);
        let replaced_expr = replace(e, &needle, replacement);
        assert_eq!(replaced_expr.to_string(), "pack(a: 1i32, b: 42i32)");
    }

    #[test]
    fn test_replace_leaves_a_lambda_alone() -> VortexResult<()> {
        // The lambda is in `list_transform`'s options, not among its children.
        let expr = list_transform(col("l"), Lambda::new(["x"], checked_add(var("x"), lit(1)))?);
        assert_eq!(replace(expr.clone(), &var("x"), lit(0)), expr);
        assert_eq!(
            replace(expr, &root(), col("s")).to_string(),
            "vortex.list.transform($.s.l, (x) -> (@x + 1i32))"
        );
        Ok(())
    }
}
