// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{VortexExpect, VortexResult};

use crate::ExprRef;
use crate::traversal::{MutNodeVisitor, Node, TransformResult, TraversalOrder};

/// Replaces all occurrences of `needle` in the expression `expr` with `replacement`.
pub fn replace(expr: ExprRef, needle: &ExprRef, replacement: ExprRef) -> ExprRef {
    let mut transform = ReplaceVisitor {
        needle,
        replacement,
    };
    expr.transform(&mut transform)
        .vortex_expect("ReplaceVisitor should not fail")
        .into_inner()
}

/// A visitor that replaces occurrences of a specific expression (`needle`) with a replacement
/// expression (`replacement`).
struct ReplaceVisitor<'a> {
    needle: &'a ExprRef,
    replacement: ExprRef,
}

impl MutNodeVisitor for ReplaceVisitor<'_> {
    type NodeTy = ExprRef;

    fn visit_down(&mut self, node: &Self::NodeTy) -> VortexResult<TraversalOrder> {
        if self.needle.eq(&node) {
            // Short-circuit traversal if the needle is found
            Ok(TraversalOrder::Skip)
        } else {
            Ok(TraversalOrder::Continue)
        }
    }

    fn visit_up(&mut self, node: Self::NodeTy) -> VortexResult<TransformResult<Self::NodeTy>> {
        if self.needle.eq(&node) {
            Ok(TransformResult::yes(self.replacement.clone()))
        } else {
            Ok(TransformResult::no(node))
        }
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
