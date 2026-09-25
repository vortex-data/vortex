// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::expr::BoundExpression;
use crate::expr::ExactBoundExpr;
use crate::expr::traversal::Node;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::NodeVisitor;
use crate::expr::traversal::TraversalOrder;

/// Boolean labels keyed by each expression node in a tree.
pub type BooleanLabels<'a, N = BoundExpression> = HashMap<&'a N, bool>;

/// Labels keyed by bound-tree identity.
pub type BoundLabels<L> = HashMap<ExactBoundExpr, L>;

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
pub fn label_tree<N, L: Clone>(
    expr: &N,
    self_label: impl Fn(&N) -> L,
    mut merge_child: impl FnMut(L, &L) -> L,
) -> HashMap<&N, L>
where
    N: Node + Eq + Hash,
{
    let mut visitor = LabelingVisitor {
        labels: Default::default(),
        self_label,
        merge_child: &mut merge_child,
    };
    expr.accept(&mut visitor)
        .vortex_expect("LabelingVisitor is infallible");
    visitor.labels
}

/// Label each node in a bound expression using identity-keyed lookups.
///
/// This avoids structurally hashing bound dtypes, which may deserialize a lazy schema.
pub fn label_bound_tree<L: Clone>(
    expr: &BoundExpression,
    self_label: impl Fn(&BoundExpression) -> L,
    mut merge_child: impl FnMut(L, &L) -> L,
) -> BoundLabels<L> {
    let mut visitor = BoundLabelingVisitor {
        labels: Default::default(),
        self_label,
        merge_child: &mut merge_child,
    };
    expr.accept(&mut visitor)
        .vortex_expect("BoundLabelingVisitor is infallible");
    visitor.labels
}

struct LabelingVisitor<'a, 'b, N, L, F, G>
where
    N: Node + Eq + Hash,
    F: Fn(&N) -> L,
    G: FnMut(L, &L) -> L,
{
    labels: HashMap<&'a N, L>,
    self_label: F,
    merge_child: &'b mut G,
}

impl<'a, 'b, N, L: Clone, F, G> NodeVisitor<'a> for LabelingVisitor<'a, 'b, N, L, F, G>
where
    N: Node + Eq + Hash,
    F: Fn(&N) -> L,
    G: FnMut(L, &L) -> L,
{
    type NodeTy = N;

    fn visit_down(&mut self, _node: &'a Self::NodeTy) -> VortexResult<TraversalOrder> {
        Ok(TraversalOrder::Continue)
    }

    fn visit_up(&mut self, node: &'a N) -> VortexResult<TraversalOrder> {
        let self_label = (self.self_label)(node);

        let final_label = node.iter_children(|children| {
            children.fold(self_label, |acc, child| {
                let child_label = self
                    .labels
                    .get(child)
                    .vortex_expect("child must have label");
                (self.merge_child)(acc, child_label)
            })
        });

        self.labels.insert(node, final_label);

        Ok(TraversalOrder::Continue)
    }
}

struct BoundLabelingVisitor<'a, L, F, G>
where
    F: Fn(&BoundExpression) -> L,
    G: FnMut(L, &L) -> L,
{
    labels: BoundLabels<L>,
    self_label: F,
    merge_child: &'a mut G,
}

impl<'node, 'visitor, L: Clone, F, G> NodeVisitor<'node> for BoundLabelingVisitor<'visitor, L, F, G>
where
    F: Fn(&BoundExpression) -> L,
    G: FnMut(L, &L) -> L,
{
    type NodeTy = BoundExpression;

    fn visit_down(&mut self, _node: &'node Self::NodeTy) -> VortexResult<TraversalOrder> {
        Ok(TraversalOrder::Continue)
    }

    fn visit_up(&mut self, node: &'node Self::NodeTy) -> VortexResult<TraversalOrder> {
        let self_label = (self.self_label)(node);
        let final_label = node.children().iter().fold(self_label, |acc, child| {
            let child_label = self
                .labels
                .get(&ExactBoundExpr(child.clone()))
                .vortex_expect("child must have label");
            (self.merge_child)(acc, child_label)
        });
        self.labels
            .insert(ExactBoundExpr(node.clone()), final_label);
        Ok(TraversalOrder::Continue)
    }
}
