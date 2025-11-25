// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_utils::aliases::hash_map::HashMap;

use super::labeling::label_tree;
use crate::expr::Expression;

/// Tracks whether an expression is null-sensitive.
///
/// An expression is null-sensitive if it or any of its children is an `is_null` operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NullSensitive {
    /// The expression or one of its children contains an `is_null` operation.
    Yes,
    /// The expression and all of its children do not contain an `is_null` operation.
    No,
}

impl NullSensitive {
    /// Combine two null sensitivity labels.
    ///
    /// Returns `Yes` if either label is `Yes`, otherwise `No`.
    pub fn combine(self, other: Self) -> Self {
        match (self, other) {
            (NullSensitive::Yes, _) | (_, NullSensitive::Yes) => NullSensitive::Yes,
            (NullSensitive::No, NullSensitive::No) => NullSensitive::No,
        }
    }
}

pub type NullSensitiveLabels<'a> = HashMap<&'a Expression, bool>;

/// Label each expression in the tree with whether it is null-sensitive.
///
/// An expression is null-sensitive if it or any of its descendants contain an `is_null` operation.
///
/// This function demonstrates the use of the general [`label_tree`] framework with:
/// - **Label Edge**: Check if the node itself is null-sensitive using [`Expression::is_null_sensitive`]
/// - **Merge Child**: Fold children labels with OR operation
pub fn label_null_sensitive(expr: &Expression) -> NullSensitiveLabels<'_> {
    label_tree(
        expr,
        // Label edge: check if this node itself is null-sensitive
        |expr| expr.is_null_sensitive(),
        // Merge child: fold children with OR (true if self OR any child is true)
        |acc, &child| acc | child,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::exprs::binary::eq;
    use crate::expr::exprs::get_item::col;
    use crate::expr::exprs::is_null::is_null;
    use crate::expr::exprs::literal::lit;

    #[test]
    fn test_null_sensitive_with_is_null() {
        // Expression: is_null($.col1)
        let expr = is_null(col("col1"));
        let labels = label_null_sensitive(&expr);

        // The root expression should be null-sensitive
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn test_null_sensitive_without_is_null() {
        // Expression: $.col1 = 5
        let expr = eq(col("col1"), lit(5));
        let labels = label_null_sensitive(&expr);

        // Since the default is conservative (true), all expressions are sensitive
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn test_null_sensitive_nested() {
        // Expression: ($.col1 = 5) = is_null($.col2)
        let left = eq(col("col1"), lit(5));
        let right = is_null(col("col2"));
        let expr = eq(left.clone(), right.clone());

        let labels = label_null_sensitive(&expr);

        // With conservative defaults, all are sensitive
        assert_eq!(labels.get(&left), Some(&true));
        assert_eq!(labels.get(&right), Some(&true));
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn test_combine() {
        assert_eq!(
            NullSensitive::Yes.combine(NullSensitive::Yes),
            NullSensitive::Yes
        );
        assert_eq!(
            NullSensitive::Yes.combine(NullSensitive::No),
            NullSensitive::Yes
        );
        assert_eq!(
            NullSensitive::No.combine(NullSensitive::Yes),
            NullSensitive::Yes
        );
        assert_eq!(
            NullSensitive::No.combine(NullSensitive::No),
            NullSensitive::No
        );
    }
}
