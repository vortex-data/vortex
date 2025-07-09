// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use vortex_error::{VortexExpect, VortexResult};
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::traversal::{Node, NodeVisitor, TraversalOrder};
use crate::{ExprRef, Identifier, VarVTable};

pub trait Annotation: Clone + Hash + Eq {}

impl<A> Annotation for A where A: Clone + Hash + Eq {}

pub type Annotations<'a, A> = HashMap<&'a ExprRef, HashSet<A>>;

/// Walk the expression tree and annotate each expression with zero or more annotations.
///
/// Returns a map of each expression to all annotations that any of its descendent (child)
/// expressions are annotated with.
pub fn descendent_annotations<A: Annotation, F>(expr: &ExprRef, annotate: F) -> Annotations<A>
where
    F: Fn(&ExprRef) -> Vec<A>,
{
    let mut visitor = AnnotationVisitor {
        annotations: Default::default(),
        annotate,
    };
    expr.accept(&mut visitor).vortex_expect("Infallible");
    visitor.annotations
}

pub fn variable_scope_annotations<A: Annotation>(
    expr: &ExprRef,
    f: impl Fn(&Identifier) -> A,
) -> Annotations<A> {
    descendent_annotations(expr, move |node| {
        if let Some(variable) = node.as_opt::<VarVTable>() {
            return vec![f(variable.var())];
        }
        vec![]
    })
}

struct AnnotationVisitor<'a, A, F> {
    annotations: Annotations<'a, A>,
    annotate: F,
}

impl<'a, A: Annotation, F> NodeVisitor<'a> for AnnotationVisitor<'a, A, F>
where
    F: Fn(&ExprRef) -> Vec<A>,
{
    type NodeTy = ExprRef;

    fn visit_down(&mut self, node: &'a Self::NodeTy) -> VortexResult<TraversalOrder> {
        let annotations = (self.annotate)(node);
        if annotations.is_empty() {
            // If the annotate fn returns empty, we do not annotate this node.
            Ok(TraversalOrder::Continue)
        } else {
            self.annotations
                .entry(node)
                .or_default()
                .extend(annotations);
            Ok(TraversalOrder::Skip)
        }
    }

    fn visit_up(&mut self, node: &'a ExprRef) -> VortexResult<TraversalOrder> {
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
