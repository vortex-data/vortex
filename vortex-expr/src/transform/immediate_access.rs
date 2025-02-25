use itertools::Itertools;
use vortex_array::aliases::hash_map::HashMap;
use vortex_array::aliases::hash_set::HashSet;
use vortex_dtype::{FieldName, StructDType};
use vortex_error::{VortexResult, vortex_err};

use crate::traversal::{Node, NodeVisitor, TraversalOrder};
use crate::{ExprRef, GetItem, Identity, Select};

pub type FieldAccesses<'a> = HashMap<&'a ExprRef, HashSet<FieldName>>;

/// For all subexpressions in an expression, find the fields that are accessed directly from the
/// scope, but not any fields in those fields
/// e.g. scope = {a: {b: .., c: ..}, d: ..}, expr = ident().a.b + ident().d accesses {a,d} (not b).
///
/// Note: This is a very naive, but simple analysis to find the fields that are accessed directly on an
/// identity node. This is combined to provide an over-approximation of the fields that are accessed
/// by an expression.
pub fn immediate_scope_accesses<'a>(
    expr: &'a ExprRef,
    scope_dtype: &'a StructDType,
) -> VortexResult<FieldAccesses<'a>> {
    ImmediateScopeAccessesAnalysis::<'a>::analyze(expr, scope_dtype)
}

/// This returns the immediate scope_access (as explained `immediate_scope_accesses`) for `expr`.
pub fn immediate_scope_access<'a>(
    expr: &'a ExprRef,
    scope_dtype: &'a StructDType,
) -> VortexResult<HashSet<FieldName>> {
    ImmediateScopeAccessesAnalysis::<'a>::analyze(expr, scope_dtype)?
        .get(expr)
        .ok_or_else(|| {
            vortex_err!("Expression missing from scope accesses, this is a internal bug")
        })
        .cloned()
}

struct ImmediateScopeAccessesAnalysis<'a> {
    sub_expressions: FieldAccesses<'a>,
    scope_dtype: &'a StructDType,
}

impl<'a> ImmediateScopeAccessesAnalysis<'a> {
    fn new(scope_dtype: &'a StructDType) -> Self {
        Self {
            sub_expressions: HashMap::new(),
            scope_dtype,
        }
    }

    fn analyze(expr: &'a ExprRef, scope_dtype: &'a StructDType) -> VortexResult<FieldAccesses<'a>> {
        let mut analysis = Self::new(scope_dtype);
        expr.accept(&mut analysis)?;
        Ok(analysis.sub_expressions)
    }
}

impl<'a> NodeVisitor<'a> for ImmediateScopeAccessesAnalysis<'a> {
    type NodeTy = ExprRef;

    fn visit_down(&mut self, node: &'a Self::NodeTy) -> VortexResult<TraversalOrder> {
        assert!(
            !node.as_any().is::<Select>(),
            "cannot analyse select, simply the expression"
        );
        if let Some(get_item) = node.as_any().downcast_ref::<GetItem>() {
            if get_item
                .child()
                .as_any()
                .downcast_ref::<Identity>()
                .is_some()
            {
                self.sub_expressions
                    .insert(node, HashSet::from_iter(vec![get_item.field().clone()]));

                return Ok(TraversalOrder::Skip);
            }
        } else if node.as_any().downcast_ref::<Identity>().is_some() {
            let st_dtype = &self.scope_dtype;
            self.sub_expressions
                .insert(node, st_dtype.names().iter().cloned().collect());
        }

        Ok(TraversalOrder::Continue)
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
