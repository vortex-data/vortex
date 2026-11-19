// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Delegation of pruning filters through nullable structs.
//!
//! A child reader can prune against its stored values when removing the parent mask can only add
//! matches. Exact filtering still applies the parent mask.

use vortex_array::expr::BoundExpression;
use vortex_array::expr::traversal::Transformed;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::mask::Mask;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_error::VortexResult;

/// Exposes stored values wherever removing the parent mask preserves every match.
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

    // Removing the mask exposes stored values, which can cause a fallible function to fail.
    let Some(scalar_fn) = expr.as_scalar() else {
        return Ok(Transformed::no(expr));
    };
    let signature = scalar_fn.signature();
    if !signature.is_infallible() {
        return Ok(Transformed::no(expr));
    }

    // AND and OR can only add matches at the top of the filter when nulls gain a value. Under
    // another function such as NOT, changing a null operand of AND or OR can remove a match.
    let in_filter_connectives = in_filter_connectives
        && matches!(expr.as_opt::<Binary>(), Some(Operator::And | Operator::Or));
    if !in_filter_connectives && !signature.is_strict() {
        return Ok(Transformed::no(expr));
    }

    let mut changed = false;
    let children = expr
        .children()
        .iter()
        .map(|child| {
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
