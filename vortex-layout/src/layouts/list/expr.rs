// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::dtype::DType;
use vortex_array::dtype::Nullability;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::BoundLambda;
use vortex_array::expr::bound::not;
use vortex_array::expr::traversal::NodeExt;
use vortex_array::expr::traversal::Transformed;
use vortex_array::expr::traversal::TraversalOrder;
use vortex_array::scalar_fn::fns::is_not_null::IsNotNull;
use vortex_array::scalar_fn::fns::is_null::IsNull;
use vortex_array::scalar_fn::fns::list_length::ListLength;
use vortex_array::scalar_fn::fns::list_transform::ListTransform;
use vortex_array::scalar_fn::fns::list_transform::ListTransformOptions;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::layouts::struct_::Item;
use crate::layouts::struct_::step_into_struct_field;

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

/// Split an expression that reads the list only through `list_transform($, λ)` into λ's body over
/// the elements and the expression over the transformed list.
///
/// The elements reader then evaluates the body, so it reads only the element fields the body uses.
/// λ's body is bound over its inputs, `{x: element}`, and stepping into `x` puts it over the
/// elements. Every use must share one λ, and λ must be infallible: the reader runs it over every
/// element between the first and last list read, including those of null and unselected lists.
pub(super) fn split_list_transform(
    expr: &BoundExpression,
) -> VortexResult<Option<(BoundExpression, BoundExpression)>> {
    let mut lambda: Option<BoundLambda> = None;
    let mut reads_list = false;
    let over_transformed = expr
        .clone()
        .transform_down(|node| {
            if let Some(ListTransformOptions::Bound(found)) = node.as_opt::<ListTransform>()
                && node.child(0).is_root()
                && found.is_infallible()
                && lambda.as_ref().is_none_or(|lambda| lambda == found)
            {
                lambda = Some(found.clone());
                return Ok(Transformed {
                    value: BoundExpression::new_root(node.dtype().clone()),
                    changed: true,
                    order: TraversalOrder::Skip,
                });
            }
            reads_list |= node.is_root();
            Ok(Transformed::no(node))
        })?
        .into_inner();

    let Some(lambda) = lambda else {
        return Ok(None);
    };
    if reads_list {
        return Ok(None);
    }

    let param = Item::Field(lambda.params().names()[0].clone());
    let element = lambda
        .params()
        .field_by_index(0)
        .vortex_expect("list_transform's lambda takes the element");
    let over_elements = step_into_struct_field(lambda.body().clone(), &param, element)?;
    Ok(Some((over_elements, over_transformed)))
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
