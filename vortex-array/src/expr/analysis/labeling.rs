// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::expr::Expression;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::NodeVisitor;
use crate::expr::traversal::TraversalOrder;

/// Label each node in an expression tree using a bottom-up traversal.
///
/// This function separates tree labeling into two distinct steps:
/// 1. **Label Self**: Compute a label for each node based only on the node itself
/// 2. **Merge Child**: Fold/accumulate labels from children into the node's self-label
///
/// The labeling process:
/// - First, `self_label` is called on the node to produce its self-label
/// - Then, for each child, `merge_child` is called with `(self_label, child_label)`
///   to fold the child label into the self_label
/// - This produces the final label for the node
///
/// # Parameters
///
/// - `expr`: The root expression to label
/// - `self_label`: Function that computes a label for a single node
/// - `merge_child`: Mutable function that folds child labels into an accumulator.
///   Takes `(self_label, child_label)` and returns the updated accumulator.
///   Called once per child, with the initial accumulator being the node's self-label.
pub fn label_tree<L: Clone>(
    expr: &Expression,
    self_label: impl Fn(&Expression) -> L,
    mut merge_child: impl FnMut(L, &L) -> L,
) -> HashMap<&Expression, L> {
    let mut visitor = LabelingVisitor {
        labels: Default::default(),
        self_label,
        merge_child: &mut merge_child,
    };
    expr.accept(&mut visitor)
        .vortex_expect("LabelingVisitor is infallible");
    visitor.labels
}

struct LabelingVisitor<'a, 'b, L, F, G>
where
    F: Fn(&Expression) -> L,
    G: FnMut(L, &L) -> L,
{
    labels: HashMap<&'a Expression, L>,
    self_label: F,
    merge_child: &'b mut G,
}

impl<'a, 'b, L: Clone, F, G> NodeVisitor<'a> for LabelingVisitor<'a, 'b, L, F, G>
where
    F: Fn(&Expression) -> L,
    G: FnMut(L, &L) -> L,
{
    type NodeTy = Expression;

    fn visit_down(&mut self, _node: &'a Self::NodeTy) -> VortexResult<TraversalOrder> {
        Ok(TraversalOrder::Continue)
    }

    fn visit_up(&mut self, node: &'a Expression) -> VortexResult<TraversalOrder> {
        let self_label = (self.self_label)(node);

        let final_label = node.children().iter().fold(self_label, |acc, child| {
            let child_label = self
                .labels
                .get(child)
                .vortex_expect("child must have label");
            (self.merge_child)(acc, child_label)
        });

        self.labels.insert(node, final_label);

        Ok(TraversalOrder::Continue)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::lit;

    #[test]
    fn test_tree_depth() {
        // Expression: $.col1 = 5
        // Tree: eq(get_item(root(), "col1"), lit(5))
        // Depth: root = 1, get_item = 2, lit = 1, eq = 3
        let expr = eq(col("col1"), lit(5));
        let depths = label_tree(
            &expr,
            |_node| 1, // Each node has depth 1 by itself
            |self_depth, child_depth| self_depth.max(*child_depth + 1),
        );

        // The root (eq) should have depth 3
        assert_eq!(depths.get(&expr), Some(&3));
    }

    #[test]
    fn test_node_count() {
        // Count total nodes in subtree (including self)
        // Tree: eq(get_item(root(), "col1"), lit(5))
        // Nodes: eq, get_item, root, lit = 4
        let expr = eq(col("col1"), lit(5));
        let counts = label_tree(
            &expr,
            |_node| 1, // Each node counts as 1
            |self_count, child_count| self_count + *child_count,
        );

        // Root should have count of 4 (eq, get_item, root, lit)
        assert_eq!(counts.get(&expr), Some(&4));
    }
}
