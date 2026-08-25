// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::expr::Expression;
use crate::expr::analysis::BooleanLabels;
use crate::expr::label_tree;

/// Label each expression with whether its entire subtree is infallible.
///
/// A subtree is infallible only when the node's scalar function and every child subtree are
/// infallible. See [`crate::scalar_fn::ScalarFnVTable::is_infallible`] for the scalar-function
/// contract.
pub fn label_infallible(expr: &Expression) -> BooleanLabels<'_> {
    label_tree(
        expr,
        |expr| match expr {
            Expression::Scalar { scalar_fn, .. } => scalar_fn.signature().is_infallible(),
            // The scope itself cannot fail.
            Expression::Root => true,
        },
        |acc, &child| acc & child,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::checked_add;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::is_null;
    use crate::expr::lit;
    use crate::expr::merge_opts;
    use crate::expr::not;
    use crate::scalar_fn::fns::merge::DuplicateHandling;

    #[test]
    fn not_is_infallible() {
        let expr = not(col("x"));
        let labels = label_infallible(&expr);
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn checked_add_defaults_to_fallible() {
        let expr = checked_add(col("a"), col("b"));
        let labels = label_infallible(&expr);
        assert_eq!(labels.get(&expr), Some(&false));
    }

    #[test]
    fn eq_is_infallible() {
        let expr = eq(col("a"), lit(5));
        let labels = label_infallible(&expr);
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn merge_with_error_handling_is_fallible() {
        let expr = merge_opts([col("a"), col("b")], DuplicateHandling::Error);
        let labels = label_infallible(&expr);
        assert_eq!(labels.get(&expr), Some(&false));
    }

    #[test]
    fn merge_with_rightmost_handling_is_infallible() {
        let expr = merge_opts([col("a"), col("b")], DuplicateHandling::RightMost);
        let labels = label_infallible(&expr);
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn nested_with_fallible_child() {
        let child = checked_add(col("a"), col("b"));
        let expr = not(child.clone());
        let labels = label_infallible(&expr);
        assert_eq!(labels.get(&child), Some(&false));
        assert_eq!(labels.get(&expr), Some(&false));
    }

    #[test]
    fn nested_without_fallible_child() {
        let child = is_null(col("x"));
        let expr = not(child.clone());
        let labels = label_infallible(&expr);
        assert_eq!(labels.get(&child), Some(&true));
        assert_eq!(labels.get(&expr), Some(&true));
    }
}
