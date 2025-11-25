// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_error::{VortexExpect, VortexResult};
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::expr::Expression;
use crate::expr::traversal::{NodeExt, NodeVisitor, TraversalOrder};

pub trait Annotation: Clone + Hash + Eq {}

impl<A> Annotation for A where A: Clone + Hash + Eq {}

pub trait AnnotationFn: Fn(&Expression) -> Vec<Self::Annotation> {
    type Annotation: Annotation;
}

impl<A, F> AnnotationFn for F
where
    A: Annotation,
    F: Fn(&Expression) -> Vec<A>,
{
    type Annotation = A;
}

pub type Annotations<'a, A> = HashMap<&'a Expression, HashSet<A>>;

/// Walk the expression tree and annotate each expression with zero or more annotations.
///
/// Returns a map of each expression to all annotations that any of its descendent (child)
/// expressions are annotated with.
///
/// This uses a specialized traversal strategy with early termination:
/// - If a node is directly annotated (non-empty), it uses only those annotations and
///   **skips traversing its children entirely**
/// - If a node is not directly annotated (empty), it traverses children and bubbles up
///   their annotations
///
/// This "skip" behavior makes this function different from [`label_tree`], which always
/// visits all nodes. Use this when you want to find the "shallowest" matches in a tree.
///
/// Note: This cannot use [`label_tree`] because the early termination (skip) requires
/// conditional traversal based on the node's direct annotations.
pub fn descendent_annotation_union_set<A: AnnotationFn>(
    expr: &Expression,
    annotate: A,
) -> Annotations<'_, A::Annotation> {
    let mut visitor = AnnotationVisitor {
        annotations: Default::default(),
        annotate,
    };
    expr.accept(&mut visitor).vortex_expect("Infallible");
    visitor.annotations
}

struct AnnotationVisitor<'a, A: AnnotationFn> {
    annotations: Annotations<'a, A::Annotation>,
    annotate: A,
}

impl<'a, A: AnnotationFn> NodeVisitor<'a> for AnnotationVisitor<'a, A> {
    type NodeTy = Expression;

    fn visit_down(&mut self, node: &'a Self::NodeTy) -> VortexResult<TraversalOrder> {
        let annotations = (self.annotate)(node);
        if annotations.is_empty() {
            // If the annotate fn returns empty, we do not annotate this node directly.
            // Continue traversing to check children.
            Ok(TraversalOrder::Continue)
        } else {
            // Node is directly annotated - store these annotations and skip children
            self.annotations
                .entry(node)
                .or_default()
                .extend(annotations);
            Ok(TraversalOrder::Skip)
        }
    }

    fn visit_up(&mut self, node: &'a Expression) -> VortexResult<TraversalOrder> {
        // Bubble up child annotations to this node
        let child_annotations = node
            .children()
            .iter()
            .filter_map(|c| self.annotations.get(c).cloned())
            .collect::<Vec<_>>();

        let annotations = self.annotations.entry(node).or_default();
        child_annotations
            .into_iter()
            .for_each(|ps| annotations.extend(ps.iter().cloned()));

        Ok(TraversalOrder::Continue)
    }
}

// Keep the old name for backwards compatibility
pub use descendent_annotation_union_set as descendent_annotations;
