// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexResult;

use crate::dtype::Nullability;
use crate::dtype::StructFields;
use crate::expr::BoundExpr;
use crate::expr::pack;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::Transformed;
use crate::expr::traversal::TraversalOrder;
use crate::expr::try_get_item;

/// Replaces all occurrences of `needle` in the expression `expr` with `replacement`.
///
/// Fallible: ancestors of a replaced node are rebuilt and re-resolve their return dtype, which
/// fails if the replacement's dtype is incompatible with the surrounding calls.
pub fn replace(
    expr: BoundExpr,
    needle: &BoundExpr,
    replacement: BoundExpr,
) -> VortexResult<BoundExpr> {
    Ok(expr
        .transform_down(|node| {
            if &node == needle {
                Ok(Transformed {
                    value: replacement.clone(),
                    // If there is a match with a needle there can be no more matches in that
                    // subtree.
                    order: TraversalOrder::Skip,
                    changed: true,
                })
            } else {
                Ok(Transformed::no(node))
            }
        })?
        .into_inner())
}

/// Expand the `root` expression with a pack of the given struct fields.
///
/// Fallible: fails if any of `fields` is absent from the root's bound scope, or if an ancestor
/// rebuild rejects the packed replacement's dtype.
pub fn replace_root_fields(expr: BoundExpr, fields: &StructFields) -> VortexResult<BoundExpr> {
    Ok(expr
        .transform_down(|node| match &node {
            BoundExpr::Root(_) => {
                let root = node.clone();
                let cols = fields
                    .names()
                    .iter()
                    .map(|name| Ok((name.clone(), try_get_item(name.clone(), root.clone())?)))
                    .collect::<VortexResult<Vec<_>>>()?;
                Ok(Transformed {
                    value: pack(cols, Nullability::NonNullable),
                    order: TraversalOrder::Skip,
                    changed: true,
                })
            }
            _ => Ok(Transformed::no(node)),
        })?
        .into_inner())
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::replace;
    use crate::dtype::Nullability::NonNullable;
    use crate::expr::get_item;
    use crate::expr::lit;
    use crate::expr::pack;

    #[test]
    fn test_replace_full_tree() -> VortexResult<()> {
        let e = get_item("b", pack([("a", lit(1)), ("b", lit(2))], NonNullable));
        let needle = get_item("b", pack([("a", lit(1)), ("b", lit(2))], NonNullable));
        let replacement = lit(42);
        let replaced_expr = replace(e, &needle, replacement.clone())?;
        assert_eq!(&replaced_expr, &replacement);
        Ok(())
    }

    #[test]
    fn test_replace_leaf() -> VortexResult<()> {
        let e = pack([("a", lit(1)), ("b", lit(2))], NonNullable);
        let needle = lit(2);
        let replacement = lit(42);
        let replaced_expr = replace(e, &needle, replacement)?;
        assert_eq!(replaced_expr.to_string(), "pack(a: 1i32, b: 42i32)");
        Ok(())
    }
}
