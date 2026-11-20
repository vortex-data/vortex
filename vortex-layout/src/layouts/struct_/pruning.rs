// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lets the classic scan use a field's zone map beneath a nullable struct.
//!
//! Removing `s`'s validity from `s.x < 10` can expose a stored value in a row that was null.
//! This can add matches, so a zone that rules out the broader filter also rules out the original.
//! Exact filtering still applies the parent mask.
//!
//! The child reader proves this with the field's own zone map. Releases up to 0.88.0 wrote zone
//! maps beneath nullable structs over the stored values, so their null counts miss the struct's
//! nulls. No file records whether a zone map excludes hidden values, so the mask must only be
//! removed where showing a hidden value cannot remove a match.

use vortex_array::expr::BoundExpression;
use vortex_array::expr::traversal::Transformed;
use vortex_array::scalar_fn::fns::between::Between;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::dynamic::DynamicComparison;
use vortex_array::scalar_fn::fns::mask::Mask;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexResult;

/// Removes masks for `parent_validity` where every original match is preserved.
///
/// Each struct reader owns the mask for its own level.
pub(super) fn unmask_pruning_filter(
    filter: BoundExpression,
    parent_validity: &BoundExpression,
) -> VortexResult<BoundExpression> {
    Ok(unmask(filter, parent_validity, true)?.into_inner())
}

fn unmask(
    expr: BoundExpression,
    parent_validity: &BoundExpression,
    in_filter_connectives: bool,
) -> VortexResult<Transformed<BoundExpression>> {
    if expr.is::<Mask>() && expr.child(1) == parent_validity {
        return Ok(Transformed::yes(expr.child(0).clone()));
    }

    let Some(scalar_fn) = expr.as_scalar() else {
        return Ok(Transformed::no(expr));
    };
    let signature = scalar_fn.signature();

    // A fallible function could error on a value previously hidden by the mask.
    if !signature.is_infallible() {
        return Ok(Transformed::no(expr));
    }

    let in_filter_connectives = in_filter_connectives
        && matches!(expr.as_opt::<Binary>(), Some(Operator::And | Operator::Or));
    let unmaskable = unmaskable_children(&expr, signature.is_strict(), in_filter_connectives);

    let mut changed = false;
    let children = expr
        .children()
        .iter()
        .enumerate()
        .map(|(idx, child)| {
            if idx >= unmaskable {
                return Ok(child.clone());
            }
            let child = unmask(child.clone(), parent_validity, in_filter_connectives)?;
            changed |= child.changed;
            Ok(child.into_inner())
        })
        .collect::<VortexResult<Vec<_>>>()?;
    if !changed {
        return Ok(Transformed::no(expr));
    }

    Ok(Transformed::yes(expr.with_children(children)?))
}

/// Returns how many leading children of `expr` can gain a stored value in a masked row without
/// removing a match of `expr`.
fn unmaskable_children(
    expr: &BoundExpression,
    is_strict: bool,
    in_filter_connectives: bool,
) -> usize {
    // A strict function propagates the null from a masked input, so exposing a value can only
    // change a null result. At the top of the filter, `AND` and `OR` preserve a true result when a
    // null operand gains a value.
    if is_strict || in_filter_connectives {
        return expr.children().len();
    }

    // `Between` is not strict, because `NULL AND false` is false for a null bound. A
    // `DynamicComparison` is not strict, because it returns its default while it has no value. A
    // null in the first child, the compared value, still gives a null result or the default.
    if expr.is::<Between>() || expr.is::<DynamicComparison>() {
        return 1;
    }

    0
}
