// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Pushes `stat(lhs <op> rhs, min | max)` down to bounds of `lhs` and `rhs`.

use vortex_error::VortexResult;

use crate::scalar_fn::ReduceNode;
use crate::scalar_fn::fns::operators::Operator;
use crate::stats::reduce::Bound;
use crate::stats::reduce::binary_node;

pub(super) fn reduce_bound<T: ReduceNode>(
    operator: Operator,
    bound: Bound,
    node: &T,
) -> VortexResult<Option<T>> {
    let lhs = node.child(0);
    let rhs = node.child(1);
    match operator {
        // Kleene `and` and `or` are increasing in both inputs.
        Operator::And | Operator::Or => monotone(operator, (&lhs, bound), (&rhs, bound)),
        // Min/max skip NaN, which comparisons do not, so float bounds prove nothing about them.
        Operator::Eq
        | Operator::NotEq
        | Operator::Gt
        | Operator::Gte
        | Operator::Lt
        | Operator::Lte
            if lhs.node_dtype()?.is_float() || rhs.node_dtype()?.is_float() =>
        {
            Ok(None)
        }
        Operator::Gt | Operator::Gte => monotone(operator, (&lhs, bound), (&rhs, bound.flip())),
        Operator::Lt | Operator::Lte => monotone(operator, (&lhs, bound.flip()), (&rhs, bound)),
        Operator::Eq | Operator::NotEq => equality(operator, bound, &lhs, &rhs),
        // Integer bounds can overflow where no row does, which errors at evaluation, so these
        // need saturating arithmetic before they are used for pruning.
        Operator::Add => monotone(operator, (&lhs, bound), (&rhs, bound)),
        Operator::Sub => monotone(operator, (&lhs, bound), (&rhs, bound.flip())),
        Operator::Mul | Operator::Div => Ok(None),
    }
}

/// `operator(bound(lhs), bound(rhs))` for an `operator` that is monotone in each input.
fn monotone<T: ReduceNode>(
    operator: Operator,
    (lhs, lhs_bound): (&T, Bound),
    (rhs, rhs_bound): (&T, Bound),
) -> VortexResult<Option<T>> {
    let (Some(lhs), Some(rhs)) = (lhs_bound.of(lhs)?, rhs_bound.of(rhs)?) else {
        return Ok(None);
    };
    binary_node(operator, lhs, rhs).map(Some)
}

/// `x = y` and `x != y` depend on whether the value ranges of `x` and `y` overlap.
fn equality<T: ReduceNode>(
    operator: Operator,
    bound: Bound,
    lhs: &T,
    rhs: &T,
) -> VortexResult<Option<T>> {
    let (Some(lhs_min), Some(lhs_max), Some(rhs_min), Some(rhs_max)) = (
        Bound::Lower.of(lhs)?,
        Bound::Upper.of(lhs)?,
        Bound::Lower.of(rhs)?,
        Bound::Upper.of(rhs)?,
    ) else {
        return Ok(None);
    };

    let (cmp, combine) = match (operator, bound) {
        // Some row may be equal only if the ranges overlap.
        (Operator::Eq, Bound::Upper) => (Operator::Lte, Operator::And),
        // Every row is equal only if both sides are the same constant.
        (Operator::Eq, Bound::Lower) => (Operator::Eq, Operator::And),
        // Some row may differ unless both sides are the same constant.
        (Operator::NotEq, Bound::Upper) => (Operator::NotEq, Operator::Or),
        // Every row differs if the ranges are disjoint.
        (Operator::NotEq, Bound::Lower) => (Operator::Gt, Operator::Or),
        _ => unreachable!("equality is only called for `=` and `!=`"),
    };

    // `cmp(lhs_min, rhs_max) <combine> cmp(rhs_min, lhs_max)`
    let left = binary_node(cmp, lhs_min, rhs_max)?;
    let right = binary_node(cmp, rhs_min, lhs_max)?;
    binary_node(combine, left, right).map(Some)
}
