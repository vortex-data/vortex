// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::expr::analysis::BooleanLabels;
use crate::expr::label_tree;
use crate::expr::Expression;

pub fn label_is_fallible(expr: &Expression) -> BooleanLabels<'_> {
    label_tree(
        expr,
        |expr| expr.signature().is_fallible(),
        |acc, &child| acc | child,
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::exprs::binary::checked_add;
    use crate::expr::exprs::binary::eq;
    use crate::expr::exprs::get_item::col;
    use crate::expr::exprs::is_null::is_null;
    use crate::expr::exprs::literal::lit;
    use crate::expr::exprs::merge::merge_opts;
    use crate::expr::exprs::merge::DuplicateHandling;
    use crate::expr::exprs::not::not;

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
