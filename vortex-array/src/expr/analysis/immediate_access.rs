// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_dtype::FieldName;
use vortex_dtype::StructFields;
use vortex_error::VortexExpect;
use vortex_utils::aliases::hash_set::HashSet;

use crate::expr::Expression;
use crate::expr::analysis::AnnotationFn;
use crate::expr::analysis::Annotations;
use crate::expr::descendent_annotations;
use crate::expr::exprs::get_item::GetItem;
use crate::expr::exprs::root::Root;
use crate::expr::exprs::select::Select;

pub type FieldAccesses<'a> = Annotations<'a, FieldName>;

/// Creates an [`AnnotationFn`] that annotates each expression node with its "free fields".
///
/// A "free field" is a top-level field from the root scope that an expression references. This is
/// useful for column pruning, where we only read the fields that an expression actually needs.
///
/// # Annotation Rules
///
/// - **[`GetItem`] on [`Root`]**: Returns `[field_name]`.
/// - **[`Root`]**: Returns all field names from `scope` (conservative over-approximation).
/// - **[`Select`]**: Returns the included field names.
/// - **Everything else**: Returns empty (annotations aggregate from children automatically).
///
/// # Example
///
/// Given `scope = {a: {b: .., c: ..}, d: ..}` and `expr = root().a.b + root().d`:
/// - `root().a` has free fields `{a}`
/// - `root().d` has free fields `{d}`
/// - The full expression has free fields `{a, d}` (not `b`, only top-level fields are tracked)
pub fn make_free_field_annotator(
    scope: &StructFields,
) -> impl AnnotationFn<Annotation = FieldName> {
    move |expr: &Expression| {
        if let Some(selection) = expr.as_opt::<Select>() {
            if expr.child(0).is::<Root>() {
                return selection
                    .normalize_to_included_fields(scope.names())
                    .vortex_expect("Select fields must be valid for scope")
                    .into_iter()
                    .collect();
            }
        } else if let Some(field_name) = expr.as_opt::<GetItem>() {
            if expr.child(0).is::<Root>() {
                return vec![field_name.clone()];
            }
        } else if expr.is::<Root>() {
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
    expr: &'a Expression,
    scope: &'a StructFields,
) -> FieldAccesses<'a> {
    descendent_annotations(expr, make_free_field_annotator(scope))
}

/// This returns the immediate scope_access (as explained `immediate_scope_accesses`) for `expr`.
pub fn immediate_scope_access<'a>(
    expr: &'a Expression,
    scope: &'a StructFields,
) -> HashSet<FieldName> {
    immediate_scope_accesses(expr, scope)
        .get(expr)
        .vortex_expect("Expression missing from scope accesses, this is a internal bug")
        .clone()
}
