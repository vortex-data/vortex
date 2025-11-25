// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_utils::aliases::hash_map::HashMap;

use super::labeling::label_tree;
use crate::expr::Expression;

pub type NullSensitiveLabels<'a> = HashMap<&'a Expression, bool>;

/// Label each expression in the tree with whether it is null-sensitive.
///
/// See [`VTable::is_null_sensitive`] for a definition of null sensitivity.
/// This function operates on a tree of expressions, not just a single expression.
pub fn label_null_sensitive(expr: &Expression) -> NullSensitiveLabels<'_> {
    label_tree(
        expr,
        |expr| expr.is_null_sensitive(),
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
        let expr = is_null(col("col1"));
        let labels = label_null_sensitive(&expr);

        // The root expression should be null-sensitive
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn test_null_sensitive_without_is_null() {
        let expr = eq(col("col1"), lit(5));
        let labels = label_null_sensitive(&expr);

        // Since the default is conservative (true), all expressions are sensitive
        assert_eq!(labels.get(&expr), Some(&true));
    }

    #[test]
    fn test_null_sensitive_nested() {
        let left = eq(col("col1"), lit(5));
        let right = is_null(col("col2"));
        let expr = eq(left.clone(), right.clone());

        let labels = label_null_sensitive(&expr);

        // With conservative defaults, all are sensitive
        assert_eq!(labels.get(&left), Some(&true));
        assert_eq!(labels.get(&right), Some(&true));
        assert_eq!(labels.get(&expr), Some(&true));
    }
}
