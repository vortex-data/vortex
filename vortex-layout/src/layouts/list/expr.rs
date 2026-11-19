// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

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

/// How a read of the list splits between its elements reader and the lists built over what that
/// reader returns.
pub(super) struct ElementsSplit {
    /// Evaluated by the elements reader, once per element.
    pub(super) elements: BoundExpression,
    /// The expression over the lists.
    pub(super) outer: BoundExpression,
    /// Whether nothing in `elements` can fail. A fallible one must not see the elements of null
    /// lists, which can hold anything.
    pub(super) infallible: bool,
}

impl ElementsSplit {
    /// Read whole elements and evaluate `expr` over the list.
    pub(super) fn whole_elements(element_dtype: &DType, expr: &BoundExpression) -> Self {
        Self {
            elements: BoundExpression::new_root(element_dtype.clone()),
            outer: expr.clone(),
            infallible: true,
        }
    }
}

/// Split an expression that reads the list only through one `list_transform($, λ)` into what the
/// elements reader evaluates and the expression over the lists of its values.
///
/// The elements reader evaluates λ's body, so it reads only the element fields the body uses.
/// The body is bound over its inputs, `{x: element}`, and stepping into `x` puts it over the
/// elements. Several distinct lambdas, or the list also read whole (anything but its validity and
/// lengths), read whole elements.
pub(super) fn split_list_transform(expr: &BoundExpression) -> VortexResult<Option<ElementsSplit>> {
    let mut lambdas = Vec::new();
    let mut whole = ListChildrenNeeded::Validity;
    let mut list_dtype = None;
    collect_list_uses(expr, &mut lambdas, &mut whole, &mut list_dtype);
    let Some(DType::List(_, nullability)) = list_dtype else {
        return Ok(None);
    };
    let [lambda] = lambdas.as_slice() else {
        return Ok(None);
    };
    if whole == ListChildrenNeeded::All {
        return Ok(None);
    }

    let param = Item::Field(lambda.params().names()[0].clone());
    let element = lambda
        .params()
        .field_by_index(0)
        .vortex_expect("list_transform's lambda takes the element");
    let elements = step_into_struct_field(lambda.body().clone(), &param, element)?;
    let lists = BoundExpression::new_root(DType::List(
        Arc::new(elements.dtype().clone()),
        nullability,
    ));

    // Every `list_transform($, λ)` is this lambda, and any other `$` is under `is_null` or
    // `list_length`, which the lists of λ's values answer alike.
    let outer = expr
        .clone()
        .transform_down(|node| {
            Ok(if list_transform_lambda(&node).is_some() || node.is_root() {
                Transformed {
                    value: lists.clone(),
                    changed: true,
                    order: TraversalOrder::Skip,
                }
            } else {
                Transformed::no(node)
            })
        })?
        .into_inner();

    Ok(Some(ElementsSplit {
        elements,
        outer,
        infallible: lambda.is_infallible(),
    }))
}

/// The bound lambda of a `list_transform($, λ)` node.
fn list_transform_lambda(node: &BoundExpression) -> Option<&BoundLambda> {
    match node.as_opt::<ListTransform>() {
        Some(ListTransformOptions::Bound(lambda)) if node.child(0).is_root() => Some(lambda),
        _ => None,
    }
}

/// Collect the distinct lambdas of `list_transform($, λ)` nodes and what the rest of the
/// expression needs of the list. `list_dtype` is set once a lambda is found.
fn collect_list_uses(
    expr: &BoundExpression,
    lambdas: &mut Vec<BoundLambda>,
    whole: &mut ListChildrenNeeded,
    list_dtype: &mut Option<DType>,
) {
    if let Some(lambda) = list_transform_lambda(expr) {
        if !lambdas.contains(lambda) {
            lambdas.push(lambda.clone());
        }
        *list_dtype = Some(expr.child(0).dtype().clone());
        return;
    }
    let needs = if is_bound_null_root(expr) {
        ListChildrenNeeded::Validity
    } else if is_bound_list_length_root(expr) {
        ListChildrenNeeded::OffsetsAndValidity
    } else if expr.is_root() {
        ListChildrenNeeded::All
    } else {
        for child in expr.children() {
            collect_list_uses(child, lambdas, whole, list_dtype);
        }
        return;
    };
    *whole = (*whole).max(needs);
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
