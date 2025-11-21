// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::{VortexExpect, VortexResult};
use vortex_utils::aliases::hash_map::HashMap;

use crate::expr::Expression;
use crate::expr::exprs::is_null::IsNull;
use crate::expr::traversal::{NodeExt, NodeVisitor, TraversalOrder};

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

pub type NullSensitiveLabels<'a> = HashMap<&'a Expression, NullSensitive>;

/// Label each expression in the tree with whether it is null-sensitive.
///
/// An expression is null-sensitive if it or any of its descendants contain an `is_null` operation.
pub fn label_null_sensitive(expr: &Expression) -> NullSensitiveLabels<'_> {
    let mut visitor = NullSensitiveVisitor {
        labels: Default::default(),
    };
    expr.accept(&mut visitor)
        .vortex_expect("NullSensitiveVisitor is infallible");
    visitor.labels
}

struct NullSensitiveVisitor<'a> {
    labels: NullSensitiveLabels<'a>,
}

impl<'a> NodeVisitor<'a> for NullSensitiveVisitor<'a> {
    type NodeTy = Expression;

    fn visit_down(&mut self, _node: &'a Self::NodeTy) -> VortexResult<TraversalOrder> {
        // Continue traversing down
        Ok(TraversalOrder::Continue)
    }

    fn visit_up(&mut self, node: &'a Expression) -> VortexResult<TraversalOrder> {
        // Check if this node is an is_null operation
        let is_null = node.is::<IsNull>();

        // Combine labels from all children
        let children_sensitive = node
            .children()
            .iter()
            .filter_map(|child| self.labels.get(child))
            .any(|&label| label == NullSensitive::Yes);

        // This node is sensitive if it's is_null or any child is sensitive
        let label = if is_null || children_sensitive {
            NullSensitive::Yes
        } else {
            NullSensitive::No
        };

        self.labels.insert(node, label);

        Ok(TraversalOrder::Continue)
    }
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
        assert_eq!(labels.get(&expr), Some(&NullSensitive::Yes));
    }

    #[test]
    fn test_null_sensitive_without_is_null() {
        // Expression: $.col1 = 5
        let expr = eq(col("col1"), lit(5));
        let labels = label_null_sensitive(&expr);

        // The root expression should not be null-sensitive
        assert_eq!(labels.get(&expr), Some(&NullSensitive::No));
    }

    #[test]
    fn test_null_sensitive_nested() {
        // Expression: ($.col1 = 5) = is_null($.col2)
        let left = eq(col("col1"), lit(5));
        let right = is_null(col("col2"));
        let expr = eq(left.clone(), right.clone());

        let labels = label_null_sensitive(&expr);

        // The left side should not be sensitive
        assert_eq!(labels.get(&left), Some(&NullSensitive::No));

        // The right side should be sensitive
        assert_eq!(labels.get(&right), Some(&NullSensitive::Yes));

        // The root should be sensitive (because right child is sensitive)
        assert_eq!(labels.get(&expr), Some(&NullSensitive::Yes));
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
