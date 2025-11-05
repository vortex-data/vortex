// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{Nullability, StructFields};
use vortex_error::{VortexExpect, VortexResult};

use crate::Expression;
use crate::exprs::get_item::col;
use crate::exprs::pack::pack;
use crate::exprs::root::root;
use crate::traversal::{NodeExt, Transformed};

/// Replaces all occurrences of `needle` in the expression `expr` with `replacement`.
pub fn replace(expr: Expression, needle: &Expression, replacement: Expression) -> Expression {
    expr.transform_up(|node| replace_transformer(node, needle, &replacement))
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

fn replace_transformer(
    node: Expression,
    needle: &Expression,
    replacement: &Expression,
) -> VortexResult<Transformed<Expression>> {
    if &node == needle {
        Ok(Transformed::yes(replacement.clone()))
    } else {
        Ok(Transformed::no(node))
    }
}

#[cfg(test)]
mod test {
    use vortex_dtype::Nullability::NonNullable;

    use super::replace;
    use crate::exprs::get_item::get_item;
    use crate::exprs::literal::lit;
    use crate::exprs::pack::pack;

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
