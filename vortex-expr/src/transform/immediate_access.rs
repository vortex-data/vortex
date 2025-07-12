// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::{FieldName, StructFields};
use vortex_error::VortexExpect;
use vortex_utils::aliases::hash_set::HashSet;

use crate::transform::annotations::{AnnotationFn, Annotations, descendent_annotations};
use crate::{ExprRef, GetItemVTable, SelectVTable, is_root};

pub type FieldAccesses<'a> = Annotations<'a, FieldName>;

/// An [`AnnotationFn`] for annotating scope accesses.
pub fn annotate_scope_access(scope: &StructFields) -> impl AnnotationFn<Annotation = FieldName> {
    move |expr: &ExprRef| {
        assert!(
            !expr.is::<SelectVTable>(),
            "cannot analyse select, simplify the expression"
        );

        if let Some(get_item) = expr.as_opt::<GetItemVTable>() {
            if is_root(get_item.child()) {
                return vec![get_item.field().clone()];
            }
        } else if is_root(expr) {
            return scope.names().iter().cloned().collect();
        }

        vec![]
    }
}

/// For all subexpressions in an expression, find the fields that are accessed directly from the
/// scope, but not any fields in those fields
/// e.g. scope = {a: {b: .., c: ..}, d: ..}, expr = root().a.b + root().d accesses {a,d} (not b).
///
/// Note: This is a very naive, but simple analysis to find the fields that are accessed directly on an
/// identity node. This is combined to provide an over-approximation of the fields that are accessed
/// by an expression.
pub fn immediate_scope_accesses<'a>(
    expr: &'a ExprRef,
    scope: &'a StructFields,
) -> FieldAccesses<'a> {
    descendent_annotations(expr, annotate_scope_access(scope))
}

/// This returns the immediate scope_access (as explained `immediate_scope_accesses`) for `expr`.
pub fn immediate_scope_access<'a>(
    expr: &'a ExprRef,
    scope: &'a StructFields,
) -> HashSet<FieldName> {
    immediate_scope_accesses(expr, scope)
        .get(expr)
        .vortex_expect("Expression missing from scope accesses, this is a internal bug")
        .clone()
}
