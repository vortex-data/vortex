// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{FieldName, StructFields};
use vortex_error::{VortexResult, vortex_err};
use vortex_utils::aliases::hash_map::HashMap;
use vortex_utils::aliases::hash_set::HashSet;

use crate::transform::access_analysis::AccessesAnalysis;
use crate::traversal::TraversalOrder;
use crate::{ExprRef, GetItem, Select, is_root};

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
    scope_dtype: &'a StructFields,
) -> VortexResult<FieldAccesses<'a>> {
    AccessesAnalysis::analyze(expr, move |node| {
        assert!(
            !node.as_any().is::<Select>(),
            "cannot analyse select, simplify the expression"
        );
        if let Some(get_item) = node.as_any().downcast_ref::<GetItem>() {
            if is_root(get_item.child()) {
                return (TraversalOrder::Skip, vec![get_item.field().clone()]);
            }
        } else if is_root(node) {
            let st_dtype = &scope_dtype;
            return (
                TraversalOrder::Skip,
                st_dtype.names().iter().cloned().collect(),
            );
        }

        (TraversalOrder::Continue, vec![])
    })
}

/// This returns the immediate scope_access (as explained `immediate_scope_accesses`) for `expr`.
pub fn immediate_scope_access<'a>(
    expr: &'a ExprRef,
    scope_dtype: &'a StructFields,
) -> VortexResult<HashSet<FieldName>> {
    immediate_scope_accesses(expr, scope_dtype)?
        .get(expr)
        .ok_or_else(|| {
            vortex_err!("Expression missing from scope accesses, this is a internal bug")
        })
        .cloned()
}
