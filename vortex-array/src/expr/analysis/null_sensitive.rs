// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_utils::aliases::hash_map::HashMap;

use super::labeling::label_tree;
use crate::expr::BoundExpr;

pub type BooleanLabels<'a> = HashMap<&'a BoundExpr, bool>;

/// Label each expression in the tree with whether it is null-sensitive.
///
/// See [`crate::scalar_fn::ScalarFnVTable::is_null_sensitive`] for a definition of null sensitivity.
/// This function operates on a tree of expressions, not just a single expression.
pub fn label_null_sensitive(expr: &BoundExpr) -> BooleanLabels<'_> {
    label_tree(
        expr,
        |expr| expr.is_null_sensitive(),
        |acc, &child| acc | child,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dtype::DType;
    use crate::dtype::Nullability::NonNullable;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::is_null;
    use crate::expr::lit;

    fn scope() -> DType {
        let i32_dtype = DType::Primitive(PType::I32, NonNullable);
        DType::Struct(
            StructFields::from_iter([("col1", i32_dtype.clone()), ("col2", i32_dtype)]),
            NonNullable,
        )
    }

    #[test]
    fn test_null_sensitive_with_is_null() {
        let dtype = scope();
        let expr = is_null(col("col1", &dtype));
        let labels = label_null_sensitive(&expr);

        // The root expression should be null-sensitive
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn test_null_sensitive_without_is_null() {
        let dtype = scope();
        let expr = eq(col("col1", &dtype), lit(5));
        let labels = label_null_sensitive(&expr);

        // Since the default is conservative (true), all expressions are sensitive
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn test_null_sensitive_nested() {
        let dtype = scope();
        let left = eq(col("col1", &dtype), lit(5));
        let right = is_null(col("col2", &dtype));
        let expr = eq(left.clone(), right.clone());

        let labels = label_null_sensitive(&expr);

        // With conservative defaults, all are sensitive
        assert_eq!(labels.get(&left), Some(&true));
        assert_eq!(labels.get(&right), Some(&true));
        assert_eq!(labels.get(&expr), Some(&true));
    }
}
