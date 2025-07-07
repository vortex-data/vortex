// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::hash::Hash;

use itertools::Itertools;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::traversal::{Node, NodeVisitor, TraversalOrder};
use crate::{ExprRef, Identifier, Var};

pub type Accesses<'a, T> = HashMap<&'a ExprRef, HashSet<T>>;

pub struct AccessesAnalysis<'a, T, F> {
    sub_expressions: Accesses<'a, T>,
    access: F,
}

impl<'a, T, F> AccessesAnalysis<'a, T, F>
where
    F: Fn(&ExprRef) -> (TraversalOrder, Vec<T>),
    T: Hash + Clone + Eq,
{
    pub(crate) fn new(access: F) -> Self {
        Self {
            sub_expressions: HashMap::new(),
            access,
        }
    }

    pub(crate) fn analyze(expr: &'a ExprRef, access: F) -> VortexResult<Accesses<'a, T>> {
        let mut analysis = Self::new(access);
        expr.accept(&mut analysis)?;
        Ok(analysis.sub_expressions)
    }
}

impl<'a, T, F> NodeVisitor<'a> for AccessesAnalysis<'a, T, F>
where
    F: Fn(&ExprRef) -> (TraversalOrder, Vec<T>),
    T: Hash + Clone + Eq,
{
    type NodeTy = ExprRef;

    fn visit_down(&mut self, node: &'a Self::NodeTy) -> VortexResult<TraversalOrder> {
        let (continue_, access) = (self.access)(node);
        self.sub_expressions
            .insert(node, HashSet::from_iter(access));
        Ok(continue_)
    }

    fn visit_up(&mut self, node: &'a ExprRef) -> VortexResult<TraversalOrder> {
        let accesses = node
            .children()
            .iter()
            .filter_map(|c| self.sub_expressions.get(c).cloned())
            .collect_vec();

        let node_accesses = self.sub_expressions.entry(node).or_default();
        accesses
            .into_iter()
            .for_each(|fields| node_accesses.extend(fields.iter().cloned()));

        Ok(TraversalOrder::Continue)
    }
}

pub fn variable_scope_accesses<T: Clone + Hash + Eq>(
    expr: &ExprRef,
    f: impl Fn(&Identifier) -> T,
) -> VortexResult<Accesses<'_, T>> {
    AccessesAnalysis::analyze(expr, move |node| {
        if let Some(variable) = node.as_any().downcast_ref::<Var>() {
            return (TraversalOrder::Skip, vec![f(variable.var())]);
        }

        (TraversalOrder::Continue, vec![])
    })
}
