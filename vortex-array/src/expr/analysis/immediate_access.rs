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
/// - **[`Select`]**: Returns the included field names if the child is [`Expression::Root`].
/// - **[`GetItem`] on the root**: Returns `[field_name]` if the child is [`Expression::Root`].
/// - **[`Expression::Root`]**: Returns all field names from `scope` (conservative
///   over-approximation).
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
    move |expr: &Expression| match expr {
        Expression::Root => scope.names().iter().cloned().collect(),
        Expression::Lambda(_) | Expression::Variable(_) | Expression::ListTransform { .. } => {
            vec![]
        }
        Expression::Scalar {
            scalar_fn,
            children,
        } => {
            if let Some(selection) = scalar_fn.as_opt::<Select>() {
                if children[0].is_root() {
                    return selection
                        .normalize_to_included_fields(scope.names())
                        .vortex_expect("Select fields must be valid for scope")
                        .into_iter()
                        .collect();
                }
            } else if let Some(field_name) = scalar_fn.as_opt::<GetItem>()
                && children[0].is_root()
            {
                return vec![field_name.clone()];
            }

            vec![]
        }
    }
}

/// Returns the free top-level fields for bound expression nodes.
pub fn make_bound_free_field_annotator(
    scope: &StructFields,
) -> impl AnnotationFn<BoundExpression, Annotation = FieldName> {
    move |expr: &BoundExpression| match expr {
        BoundExpression::Root { .. } => scope.names().iter().cloned().collect(),
        BoundExpression::Lambda(_)
        | BoundExpression::Variable(_)
        | BoundExpression::ListTransform { .. } => vec![],
        BoundExpression::Scalar {
            scalar_fn,
            children,
            ..
        } => {
            if let Some(selection) = scalar_fn.as_opt::<Select>() {
                if children[0].is_root() {
                    return selection
                        .normalize_to_included_fields(scope.names())
                        .vortex_expect("Select fields must be valid for scope")
                        .into_iter()
                        .collect();
                }
            } else if let Some(field_name) = scalar_fn.as_opt::<GetItem>()
                && children[0].is_root()
            {
                return vec![field_name.clone()];
            }

            vec![]
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::expr::Scope;
    use crate::expr::Variable;
    use crate::expr::var;

    #[test]
    fn variables_do_not_access_root_fields() -> VortexResult<()> {
        let fields = StructFields::from_iter([("a", DType::Null)]);
        let expression = var("value");

        assert!(make_free_field_annotator(&fields)(&expression).is_empty());

        let root_dtype = DType::Struct(fields.clone(), Nullability::NonNullable);
        let variable_dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let scope =
            Scope::new(root_dtype).with_bindings([(Variable::new("value"), variable_dtype)])?;
        let bound = expression.bind_scope(&scope)?;
        assert!(make_bound_free_field_annotator(&fields)(&bound).is_empty());
        Ok(())
    }
}
