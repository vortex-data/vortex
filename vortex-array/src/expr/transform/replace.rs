// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::expr::Expression;
use crate::expr::exprs::get_item::col;
use crate::expr::exprs::pack::pack;
use crate::expr::exprs::root::root;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::Transformed;
use crate::expr::traversal::TraversalOrder;

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

/// Expand the `root` expression with a pack of the given struct fields.
pub fn replace_root_fields(expr: Expression, fields: &StructFields) -> Expression {
    replace(
        expr,
        &root(),
        pack(
            fields
                .names()
                .iter()
                .map(|name| (name.clone(), col(name.clone()))),
            Nullability::NonNullable,
        ),
    )
}

#[cfg(test)]
mod test {
    use super::replace;
    use crate::dtype::Nullability::NonNullable;
    use crate::expr::exprs::get_item::get_item;
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::pack::pack;

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
}
