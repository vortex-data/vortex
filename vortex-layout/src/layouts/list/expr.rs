// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::dtype::FieldName;
use vortex_array::dtype::Nullability;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::not;
use vortex_array::scalar_fn::fns::is_not_null::IsNotNull;
use vortex_array::scalar_fn::fns::is_null::IsNull;
use vortex_array::scalar_fn::fns::list_get_item::ListGetItem;
use vortex_array::scalar_fn::fns::list_length::ListLength;
use vortex_error::VortexResult;

/// The minimal set of list children an expression needs for evaluation.
///
/// For example:
///     - `is_null(root())` only needs the validity child.
///     - `list_length(root())` only needs the offsets and validity children.
///     - `root()` needs elements, offsets, and validity children.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(super) enum ListChildrenNeeded {
    /// Only the validity child is needed (`is_null` / `is_not_null`).
    Validity,
    /// Only the offsets and validity children are needed (`list_length`).
    OffsetsAndValidity,
    /// All children are needed.
    All,
}

/// The minimal set of list children needed to evaluate a bound expression.
pub(super) fn get_necessary_bound_list_children(expr: &BoundExpression) -> ListChildrenNeeded {
    if is_bound_null_root(expr) {
        return ListChildrenNeeded::Validity;
    }

    if is_bound_list_length_root(expr) {
        return ListChildrenNeeded::OffsetsAndValidity;
    }

    if expr.is_root() {
        return ListChildrenNeeded::All;
    }

    expr.children()
        .iter()
        .map(get_necessary_bound_list_children)
        .max()
        .unwrap_or(ListChildrenNeeded::Validity)
}

fn is_bound_null_root(expr: &BoundExpression) -> bool {
    (expr.as_scalar().is_some_and(|f| f.is::<IsNull>())
        || expr.as_scalar().is_some_and(|f| f.is::<IsNotNull>()))
        && expr.children().len() == 1
        && expr.children()[0].is_root()
}

fn is_bound_list_length_root(expr: &BoundExpression) -> bool {
    expr.as_scalar().is_some_and(|f| f.is::<ListLength>())
        && expr.children().len() == 1
        && expr.children()[0].is_root()
}

/// Rewrite a validity-class expression so it can be evaluated against the list's validity bool
/// array (`true` == valid row): `is_not_null(root())` becomes `root()` and `is_null(root())`
/// becomes `not(root())`. All other nodes are rebuilt with rewritten children.
pub(super) fn rewrite_validity_expr(expr: &BoundExpression) -> VortexResult<BoundExpression> {
    let validity_dtype = DType::Bool(Nullability::NonNullable);
    rewrite_validity_expr_with_root(expr, &validity_dtype)
}

fn rewrite_validity_expr_with_root(
    expr: &BoundExpression,
    root_dtype: &DType,
) -> VortexResult<BoundExpression> {
    if expr.as_scalar().is_some_and(|f| f.is::<IsNotNull>())
        && expr.children().len() == 1
        && expr.children()[0].is_root()
    {
        return Ok(BoundExpression::new_root(root_dtype.clone()));
    }
    if expr.as_scalar().is_some_and(|f| f.is::<IsNull>())
        && expr.children().len() == 1
        && expr.children()[0].is_root()
    {
        return Ok(not(BoundExpression::new_root(root_dtype.clone())));
    }
    if expr.is_root() {
        return Ok(BoundExpression::new_root(root_dtype.clone()));
    }

    let children = expr
        .children()
        .iter()
        .map(|child| rewrite_validity_expr_with_root(child, root_dtype))
        .collect::<VortexResult<Vec<_>>>()?;
    expr.clone().with_children(children)
}

/// How an expression uses the list root, while scanning for a single-field element projection.
enum ElementFieldUse {
    /// No reference to the root anywhere in the (sub)expression.
    NoRoot,
    /// Every root reference is `list_get_item(field, root())` with this one field.
    Field(FieldName),
    /// The root is used some other way (bare root, another field, `is_null`, `list_length`, …).
    Mixed,
}

/// Returns the single struct field projected out of the list elements, iff *every* root
/// reference in `expr` has the shape `list_get_item(field, root())` with the same field.
///
/// Such an expression only needs that one field of the elements child: the caller can push
/// `get_item(field, ·)` into the elements read and rewrite this expression with
/// [`rewrite_element_projection_expr`] to run against the narrowed list.
pub(super) fn extract_element_projection(expr: &BoundExpression) -> Option<FieldName> {
    match element_field_use(expr) {
        ElementFieldUse::Field(field) => Some(field),
        ElementFieldUse::NoRoot | ElementFieldUse::Mixed => None,
    }
}

fn element_field_use(expr: &BoundExpression) -> ElementFieldUse {
    if is_bound_element_projection(expr) {
        return ElementFieldUse::Field(expr.as_::<ListGetItem>().clone());
    }
    if expr.is_root() {
        return ElementFieldUse::Mixed;
    }

    let mut acc = ElementFieldUse::NoRoot;
    for child in expr.children() {
        acc = match (acc, element_field_use(child)) {
            (ElementFieldUse::NoRoot, child_use) => child_use,
            (acc, ElementFieldUse::NoRoot) => acc,
            (ElementFieldUse::Field(a), ElementFieldUse::Field(b)) if a == b => {
                ElementFieldUse::Field(a)
            }
            _ => return ElementFieldUse::Mixed,
        };
    }
    acc
}

fn is_bound_element_projection(expr: &BoundExpression) -> bool {
    expr.as_opt::<ListGetItem>().is_some()
        && expr.children().len() == 1
        && expr.children()[0].is_root()
}

/// Rewrite an element-projection expression so it can be evaluated against the narrowed list
/// (whose elements are already the projected field): `list_get_item(field, root())` becomes
/// `root()` bound to `new_root_dtype`. All other nodes are rebuilt with rewritten children.
pub(super) fn rewrite_element_projection_expr(
    expr: &BoundExpression,
    new_root_dtype: &DType,
) -> VortexResult<BoundExpression> {
    if is_bound_element_projection(expr) {
        debug_assert_eq!(
            expr.dtype(),
            new_root_dtype,
            "narrowed list dtype must match the list_get_item node it replaces"
        );
        return Ok(BoundExpression::new_root(new_root_dtype.clone()));
    }

    let children = expr
        .children()
        .iter()
        .map(|child| rewrite_element_projection_expr(child, new_root_dtype))
        .collect::<VortexResult<Vec<_>>>()?;
    expr.clone().with_children(children)
}

/// Rewrite an offsets-class expression so it can be evaluated against an array of list lengths.
/// `list_length(root())` becomes `root()`. Other references to `root()` are left intact: for
/// offsets-class expressions they can only be validity checks, and the lengths array carries the
/// same validity as the original list.
pub(super) fn rewrite_offsets_expr(
    expr: &BoundExpression,
    lengths_dtype: &DType,
) -> VortexResult<BoundExpression> {
    if is_bound_list_length_root(expr) || expr.is_root() {
        return Ok(BoundExpression::new_root(lengths_dtype.clone()));
    }

    let children = expr
        .children()
        .iter()
        .map(|child| rewrite_offsets_expr(child, lengths_dtype))
        .collect::<VortexResult<Vec<_>>>()?;
    expr.clone().with_children(children)
}
