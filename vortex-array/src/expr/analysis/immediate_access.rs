// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_error::VortexExpect;

use crate::dtype::FieldName;
use crate::dtype::StructFields;
use crate::expr::BoundExpression;
use crate::expr::Expression;
use crate::expr::analysis::AnnotationFn;
use crate::scalar_fn::fns::get_item::GetItem;
use crate::scalar_fn::fns::select::Select;

/// Returns the "free fields" for this expression node.
///
/// A "free field" is a top-level field from the root scope that this expression references—not
/// nested fields within those top-level fields. For example, `root().a.b` has free field `{a}`,
/// not `{b}`, because `a` is the top-level field being accessed from root.
///
/// The term "free" is borrowed from PL theory's "free variables"—variables that reference an
/// outer scope rather than being introduced locally.
///
/// This is useful for column pruning, where we only need to read the top-level fields that an
/// expression actually touches.
///
/// # Annotation Rules
///
/// - **[`Select`]**: Returns the included field names if the child is [`Root`].
/// - **[`GetItem`] on [`Root`]**: Returns `[field_name]` if the child is [`Root`].
/// - **[`Root`]**: Returns all field names from `scope` (conservative over-approximation).
/// - **Everything else**: Returns empty (annotations aggregate from children automatically).
///
/// # Example
///
/// Given `scope = {a: {b: .., c: ..}, d: ..}` and `expr = root().a.b + root().d`:
/// - `root().a` has free fields `{a}`.
/// - `root().d` has free fields `{d}`.
/// - The full expression has free fields `{a, d}` (not `b`, only top-level fields are tracked).
pub fn make_free_field_annotator(
    scope: &StructFields,
) -> impl AnnotationFn<Expression, Annotation = FieldName> {
    move |expr: &Expression| {
        if let Some(selection) = expr.as_opt::<Select>() {
            if expr.child(0).is_root() {
                return selection
                    .normalize_to_included_fields(scope.names())
                    .vortex_expect("Select fields must be valid for scope")
                    .into_iter()
                    .collect();
            }
        } else if let Some(field_name) = expr.as_opt::<GetItem>() {
            if expr.child(0).is_root() {
                return vec![field_name.clone()];
            }
        } else if expr.is_root() {
            return scope.names().iter().cloned().collect();
        }

        vec![]
    }
}

/// Returns the free top-level fields for bound expression nodes.
pub fn make_bound_free_field_annotator(
    scope: &StructFields,
) -> impl AnnotationFn<BoundExpression, Annotation = FieldName> {
    move |expr: &BoundExpression| {
        let Some(scalar_fn) = expr.as_scalar() else {
            // Only the scope root reads every field. A variable resolves against a frame, so it
            // reads none of them, and saying otherwise would defeat column pruning.
            return if expr.is_root() {
                scope.names().iter().cloned().collect()
            } else {
                vec![]
            };
        };

        if let Some(selection) = scalar_fn.as_opt::<Select>() {
            if expr.children()[0].is_root() {
                return selection
                    .normalize_to_included_fields(scope.names())
                    .vortex_expect("Select fields must be valid for scope")
                    .into_iter()
                    .collect();
            }
        } else if let Some(field_name) = scalar_fn.as_opt::<GetItem>()
            && expr.children()[0].is_root()
        {
            return vec![field_name.clone()];
        }

        vec![]
    }
}

#[cfg(test)]
mod variable_tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::Scope;
    use crate::expr::lambda;
    use crate::expr::test_harness::struct_dtype;
    use crate::expr::var;

    /// Only the scope root reads every field. A variable resolves against a frame, so treating it
    /// like a root would mark a lambda body as reading the whole struct and defeat column pruning.
    #[test]
    fn a_bound_variable_accesses_no_root_fields() -> VortexResult<()> {
        let scope_dtype = struct_dtype();
        let fields = scope_dtype
            .as_struct_fields_opt()
            .vortex_expect("test scope is a struct");

        let bound = lambda(["x"], var("x")).bind(
            &Scope::new(scope_dtype.clone()),
            [DType::Primitive(PType::I32, Nullability::NonNullable)],
        )?;

        let annotator = make_bound_free_field_annotator(fields);
        assert!(
            annotator(bound.body()).is_empty(),
            "a variable should read no root fields"
        );
        Ok(())
    }
}
