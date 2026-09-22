// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::expr::BoundExpression;
use crate::expr::ExactBoundExpr;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::NodeVisitor;
use crate::expr::traversal::TraversalOrder;

/// A value that can annotate an expression node.
pub trait Annotation: Clone + Hash + Eq {}

impl<A> Annotation for A where A: Clone + Hash + Eq {}

/// Computes annotations for an expression node, defaulting to bound expressions.
pub trait AnnotationFn<N = BoundExpression>: Fn(&N) -> Vec<Self::Annotation> {
    /// The annotation attached to a node.
    type Annotation: Annotation;
}

impl<N, A, F> AnnotationFn<N> for F
where
    A: Annotation,
    F: Fn(&N) -> Vec<A>,
{
    type Annotation = A;
}

/// Annotations keyed by bound-tree identity.
///
/// Identity keys avoid structurally hashing every node's dtype. That matters when a bound root
/// carries a lazy schema whose structural hash would deserialize every field.
pub type BoundAnnotations<A> = HashMap<ExactBoundExpr, HashSet<A>>;

/// Annotate a bound expression and propagate each annotation to its ancestors.
///
/// This uses [`ExactBoundExpr`] keys to preserve the cheap identity semantics of a bound tree.
pub fn descendent_bound_annotations<A>(
    expr: &BoundExpression,
    annotate: A,
) -> BoundAnnotations<A::Annotation>
where
    A: AnnotationFn<BoundExpression>,
{
    bound_annotations(expr, annotate, true)
}

/// Annotate each bound-expression node without propagating annotations to its ancestors.
///
/// The returned map uses [`ExactBoundExpr`] keys so lookups do not structurally hash node dtypes.
pub fn direct_bound_annotations<A>(
    expr: &BoundExpression,
    annotate: A,
) -> BoundAnnotations<A::Annotation>
where
    A: AnnotationFn<BoundExpression>,
{
    bound_annotations(expr, annotate, false)
}

fn bound_annotations<A>(
    expr: &BoundExpression,
    annotate: A,
    propagate_up: bool,
) -> BoundAnnotations<A::Annotation>
where
    A: AnnotationFn<BoundExpression>,
{
    let mut visitor = BoundAnnotationVisitor {
        annotations: Default::default(),
        annotate,
        propagate_up,
    };
    expr.accept(&mut visitor).vortex_expect("Infallible");
    visitor.annotations
}

struct BoundAnnotationVisitor<A>
where
    A: AnnotationFn<BoundExpression>,
{
    annotations: BoundAnnotations<A::Annotation>,
    annotate: A,
    propagate_up: bool,
}

impl<'a, A> NodeVisitor<'a> for BoundAnnotationVisitor<A>
where
    A: AnnotationFn<BoundExpression>,
{
    type NodeTy = BoundExpression;

    fn visit_down(&mut self, node: &'a Self::NodeTy) -> VortexResult<TraversalOrder> {
        let annotations = (self.annotate)(node);
        if annotations.is_empty() {
            return Ok(TraversalOrder::Continue);
        }

        self.annotations
            .entry(ExactBoundExpr(node.clone()))
            .or_default()
            .extend(annotations);
        Ok(TraversalOrder::Skip)
    }

    fn visit_up(&mut self, node: &'a Self::NodeTy) -> VortexResult<TraversalOrder> {
        if !self.propagate_up {
            return Ok(TraversalOrder::Continue);
        }

        let child_annotations = node
            .children()
            .iter()
            .filter_map(|child| {
                self.annotations
                    .get(&ExactBoundExpr(child.clone()))
                    .cloned()
            })
            .collect::<Vec<_>>();
        let annotations = self
            .annotations
            .entry(ExactBoundExpr(node.clone()))
            .or_default();
        child_annotations
            .into_iter()
            .for_each(|child| annotations.extend(child));

        Ok(TraversalOrder::Continue)
    }
}
