// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::expr::Expression;
use crate::expr::analysis::BooleanLabels;
use crate::expr::label_tree;

pub fn label_is_fallible(expr: &Expression) -> BooleanLabels<'_> {
    label_tree(
        expr,
        |expr| match expr {
            Expression::Scalar { scalar_fn, .. } => scalar_fn.signature().is_fallible(),
            // These add no fallibility of their own. Note this is the *self* label: a lambda's
            // body is one of its children, so the folded label at a lambda node is the body's
            // fallibility. A higher-order function therefore picks the body up through the
            // ordinary fold instead of walking it by hand.
            Expression::Root | Expression::Variable(_) | Expression::Lambda(_) => false,
        },
        |acc, &child| acc | child,
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
    fn not_is_not_fallible() {
        let expr = not(col("x"));
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&expr), Some(&false));
    }

    #[test]
    fn checked_add_defaults_to_fallible() {
        let expr = checked_add(col("a"), col("b"));
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn eq_not_fallible() {
        let expr = eq(col("a"), lit(5));
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&expr), Some(&false));
    }

    #[test]
    fn merge_with_error_handling_is_fallible() {
        let expr = merge_opts([col("a"), col("b")], DuplicateHandling::Error);
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn merge_with_rightmost_handling_is_not_fallible() {
        let expr = merge_opts([col("a"), col("b")], DuplicateHandling::RightMost);
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&expr), Some(&false));
    }

    #[test]
    fn nested_with_fallible_child() {
        let child = checked_add(col("a"), col("b"));
        let expr = not(child.clone());
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&child), Some(&true));
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn nested_without_fallible_child() {
        let child = is_null(col("x"));
        let expr = not(child.clone());
        let labels = label_is_fallible(&expr);
        assert_eq!(labels.get(&child), Some(&false));
        assert_eq!(labels.get(&expr), Some(&false));
    }
}

#[cfg(test)]
mod lambda_tests {
    use super::*;
    use crate::expr::checked_add;
    use crate::expr::lambda;
    use crate::expr::lit;
    use crate::expr::var;

    /// A lambda contributes no fallibility of its own, but its body is one of its children, so the
    /// label at the lambda node is the body's. That is what lets a future higher-order function
    /// pick the body up through the ordinary fold rather than walking it by hand.
    #[test]
    fn a_lambdas_label_is_its_bodys_fallibility() {
        let fallible = Expression::from(lambda(["x"], checked_add(var("x"), lit(1i32))));
        assert_eq!(label_is_fallible(&fallible).get(&fallible), Some(&true));

        let infallible = Expression::from(lambda(["x"], var("x")));
        assert_eq!(
            label_is_fallible(&infallible).get(&infallible),
            Some(&false)
        );
    }
}
