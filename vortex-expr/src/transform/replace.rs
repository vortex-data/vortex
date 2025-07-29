// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{Nullability, StructFields};
use vortex_error::{VortexExpect, VortexResult};

use crate::traversal::{Node, Transformed};
use crate::{ExprRef, col, pack, root};

/// Replaces all occurrences of `needle` in the expression `expr` with `replacement`.
pub fn replace(expr: ExprRef, needle: &ExprRef, replacement: ExprRef) -> ExprRef {
    expr.transform_up(|node| replace_transformer(node, needle, &replacement))
        .vortex_expect("ReplaceVisitor should not fail")
        .into_inner()
}

/// Expand the `root` expression with a pack of the given struct fields.
pub fn replace_root_fields(expr: ExprRef, fields: &StructFields) -> ExprRef {
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
    node: ExprRef,
    needle: &ExprRef,
    replacement: &ExprRef,
) -> VortexResult<Transformed<ExprRef>> {
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
    use crate::{get_item, lit, pack};

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
        let replaced_expr = replace(e, &needle, replacement.clone());
        assert_eq!(replaced_expr.to_string(), "pack(a: 1i32, b: 42i32)");
    }
}
