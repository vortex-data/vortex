// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use super::BooleanLabels;
use super::labeling::label_tree;
use crate::expr::Expression;

/// Label each expression with whether its entire subtree is strict.
///
/// A subtree is strict only when the node's scalar function and every child subtree are strict.
/// See [`crate::scalar_fn::ScalarFnVTable::is_strict`] for the scalar-function contract.
pub fn label_strict(expr: &Expression) -> BooleanLabels<'_> {
    label_tree(
        expr,
        |expr| match expr {
            Expression::Scalar { scalar_fn, .. } => scalar_fn.signature().is_strict(),
            // Vacuously strict.
            Expression::Root => true,
        },
        |acc, &child| acc & child,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::is_null;
    use crate::expr::lit;

    #[test]
    fn test_non_strict_with_is_null() {
        let expr = is_null(col("col1"));
        let labels = label_strict(&expr);

        assert_eq!(labels.get(&expr), Some(&false));
    }

    #[test]
    fn test_strict_expression() {
        let expr = eq(lit(4), lit(5));
        let labels = label_strict(&expr);

        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn test_non_strict_child_makes_parent_subtree_non_strict() {
        let left = eq(lit(4), lit(5));
        let right = is_null(col("col2"));
        let expr = eq(left.clone(), right.clone());

        let labels = label_strict(&expr);

        assert_eq!(labels.get(&left), Some(&true));
        assert_eq!(labels.get(&right), Some(&false));
        assert_eq!(labels.get(&expr), Some(&false));
    }
}
