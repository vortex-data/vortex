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

/// How an expression uses the list root, while scanning for an element-projection chain.
enum ElementFieldUse {
    /// No reference to the root anywhere in the (sub)expression.
    NoRoot,
    /// Every root reference is the same `list_get_item(fk, … list_get_item(f1, root()) …)`
    /// tower; the path is innermost-first (`[f1, …, fk]`).
    Path(Vec<FieldName>),
    /// The root is used some other way (bare root, differing chains, `is_null`, `list_length`, …).
    Mixed,
}

/// Returns the field path projected out of the list elements, iff *every* root reference in
/// `expr` is the same tower of `list_get_item` calls over `root()`. The path is returned
/// innermost-first, i.e. `list_get_item("b", list_get_item("a", root()))` yields `[a, b]`.
///
/// Such an expression only needs that leaf path of the elements child: the caller can push the
/// equivalent element-wise projection into the elements read and rewrite this expression with
/// [`rewrite_element_projection_expr`] to run against the narrowed list.
pub(super) fn extract_element_projection(expr: &BoundExpression) -> Option<Vec<FieldName>> {
    match element_field_use(expr) {
        ElementFieldUse::Path(path) => Some(path),
        ElementFieldUse::NoRoot | ElementFieldUse::Mixed => None,
    }
}

fn element_field_use(expr: &BoundExpression) -> ElementFieldUse {
    if let Some(path) = chain_path(expr) {
        return ElementFieldUse::Path(path);
    }
    if expr.is_root() {
        return ElementFieldUse::Mixed;
    }

    let mut acc = ElementFieldUse::NoRoot;
    for child in expr.children() {
        acc = match (acc, element_field_use(child)) {
            (ElementFieldUse::NoRoot, child_use) => child_use,
            (acc, ElementFieldUse::NoRoot) => acc,
            (ElementFieldUse::Path(a), ElementFieldUse::Path(b)) if a == b => {
                ElementFieldUse::Path(a)
            }
            _ => return ElementFieldUse::Mixed,
        };
    }
    acc
}

/// If `expr` is a tower of `list_get_item` nodes whose innermost child is `root()`, return the
/// projected field path innermost-first; otherwise `None`.
fn chain_path(expr: &BoundExpression) -> Option<Vec<FieldName>> {
    let field = expr.as_opt::<ListGetItem>()?;
    if expr.children().len() != 1 {
        return None;
    }
    let child = &expr.children()[0];
    if child.is_root() {
        return Some(vec![field.clone()]);
    }
    let mut path = chain_path(child)?;
    path.push(field.clone());
    Some(path)
}

/// Rewrite an element-projection expression so it can be evaluated against the narrowed list
/// (whose elements are already the projected leaf path): every `list_get_item` tower matching
/// `path` becomes `root()` bound to `new_root_dtype`. All other nodes are rebuilt with
/// rewritten children.
pub(super) fn rewrite_element_projection_expr(
    expr: &BoundExpression,
    path: &[FieldName],
    new_root_dtype: &DType,
) -> VortexResult<BoundExpression> {
    if chain_path(expr).is_some_and(|p| p == path) {
        debug_assert_eq!(
            expr.dtype(),
            new_root_dtype,
            "narrowed list dtype must match the list_get_item tower it replaces"
        );
        return Ok(BoundExpression::new_root(new_root_dtype.clone()));
    }

    let children = expr
        .children()
        .iter()
        .map(|child| rewrite_element_projection_expr(child, path, new_root_dtype))
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
