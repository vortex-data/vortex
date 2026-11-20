// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Uses stored zone statistics to prove that filters on a nullable [`Pack`](super::Pack)
//! cannot match.
//!
//! For an integer `s.x`, `min(s.x) >= 10` proves that `s.x < 10` has no matches. A value hidden
//! by `s` can lower the stored minimum, making the proof harder to satisfy. When the stored
//! minimum is still at least 10, the scan can skip the zone.
//!
//! Every zone map beneath a nullable struct must be treated as covering hidden values. Releases
//! up to 0.88.0 wrote them over the stored values, so their null counts miss the struct's nulls,
//! and no file records whether a zone map excludes hidden values.

use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::all_non_nan::AllNonNan;
use vortex_array::aggregate_fn::fns::bounded_max::BoundedMax;
use vortex_array::aggregate_fn::fns::bounded_min::BoundedMin;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::Scalar;
use vortex_array::scalar_fn::fns::binary::Binary;
use vortex_array::scalar_fn::fns::literal::Literal;
use vortex_array::scalar_fn::fns::mask::Mask;
use vortex_array::scalar_fn::fns::operators::Operator;
use vortex_array::scalar_fn::fns::stat::StatFn;
use vortex_error::VortexResult;
use vortex_expr::stats;

/// Weakens a proof over masked fields so it can use their stored statistics.
///
/// Every true result still proves that a zone can be skipped.
pub(super) fn weaken_pruning_proof(proof: BoundExpression) -> VortexResult<BoundExpression> {
    if matches!(proof.as_opt::<Binary>(), Some(Operator::And | Operator::Or)) {
        let left = weaken_pruning_proof(proof.child(0).clone())?;
        let right = weaken_pruning_proof(proof.child(1).clone())?;
        return proof.with_children([left, right]);
    }

    if !proof.contains::<Mask>()? {
        return Ok(proof);
    }

    if let Some(physical) = unmask_proof_clause(&proof)? {
        return Ok(physical);
    }

    // Drop the whole clause. Putting a null placeholder inside `is_null(stat)` would turn
    // an unsupported statistic into a true proof.
    Ok(if proof.dtype().is_nullable() {
        bound::lit(Scalar::null(proof.dtype().clone()))
    } else {
        bound::lit(false)
    })
}

fn unmask_proof_clause(proof: &BoundExpression) -> VortexResult<Option<BoundExpression>> {
    if let Some(input) = unmasked_stat_input(proof, |agg| agg.is::<AllNonNan>())? {
        return Ok(Some(stats::bound::all_non_nan(input)));
    }

    let Some(operator) = proof.as_opt::<Binary>() else {
        return Ok(None);
    };
    let (left, right) = (proof.child(0), proof.child(1));

    if *operator == Operator::Eq {
        let (zero, count) = if left.is::<Literal>() {
            (left, right)
        } else {
            (right, left)
        };
        if zero.as_opt::<Literal>().and_then(Scalar::is_zero) != Some(true) {
            return Ok(None);
        }

        // A stored count of zero proves all visible values are non-NaN. A positive count
        // can come from hidden rows, so it gives no proof about the visible values.
        let input = unmasked_stat_input(count, |agg| agg.is::<NanCount>())?;
        return Ok(input.map(stats::bound::all_non_nan));
    }

    // For `<`, hidden values can raise the left upper bound or lower the right lower bound.
    // Both changes make the proof harder to satisfy.
    let (left_bound, right_bound) = match operator {
        Operator::Lt | Operator::Lte => (StatBound::Upper, StatBound::Lower),
        Operator::Gt | Operator::Gte => (StatBound::Lower, StatBound::Upper),
        _ => return Ok(None),
    };

    // Only move bounds used directly by a comparison. A fallible function between a bound and
    // the comparison could fail on a stored value hidden by the parent.
    let Some(left) = comparison_operand(left, left_bound)? else {
        return Ok(None);
    };
    let Some(right) = comparison_operand(right, right_bound)? else {
        return Ok(None);
    };

    Ok(Some(proof.clone().with_children([left, right])?))
}

#[derive(Clone, Copy)]
enum StatBound {
    Lower,
    Upper,
}

impl StatBound {
    fn matches(self, aggregate: &AggregateFnRef) -> bool {
        let stat = Stat::from_aggregate_fn(aggregate);
        match self {
            Self::Lower => stat == Some(Stat::Min) || aggregate.is::<BoundedMin>(),
            Self::Upper => stat == Some(Stat::Max) || aggregate.is::<BoundedMax>(),
        }
    }
}

fn comparison_operand(
    expr: &BoundExpression,
    bound: StatBound,
) -> VortexResult<Option<BoundExpression>> {
    if !expr.contains::<Mask>()? {
        return Ok(Some(expr.clone()));
    }

    let Some(input) = unmasked_stat_input(expr, |agg| bound.matches(agg))? else {
        return Ok(None);
    };
    let unmasked = expr.clone().with_children([input])?;

    Ok((unmasked.dtype() == expr.dtype()).then_some(unmasked))
}

fn unmasked_stat_input(
    stat: &BoundExpression,
    allow: impl FnOnce(&AggregateFnRef) -> bool,
) -> VortexResult<Option<BoundExpression>> {
    let Some(opts) = stat.as_opt::<StatFn>() else {
        return Ok(None);
    };
    let mut input = stat.child(0);
    if !input.is::<Mask>() || !allow(opts.aggregate_fn()) {
        return Ok(None);
    }

    while input.is::<Mask>() {
        input = input.child(0);
    }

    Ok((!input.contains::<Mask>()?).then(|| input.clone()))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::NumericalAggregateOpts;
    use vortex_array::aggregate_fn::fns::max::Max;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::expr::Expression;
    use vortex_array::expr::cast;
    use vortex_array::expr::eq;
    use vortex_array::expr::gt;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt_eq;
    use vortex_array::expr::mask;
    use vortex_array::expr::root;
    use vortex_array::scalar::Scalar;
    use vortex_error::VortexResult;
    use vortex_expr::stats;

    use super::weaken_pruning_proof;

    fn masked_input() -> Expression {
        mask(root(), is_not_null(root()))
    }

    fn max(input: Expression) -> Expression {
        stats::stat(input, Max.bind(NumericalAggregateOpts::skip_nans()))
    }

    fn unknown() -> Expression {
        lit(Scalar::null(DType::Bool(Nullability::Nullable)))
    }

    #[rstest]
    #[case::widening_weakens_proof(
        lt_eq(max(masked_input()), lit(5.0f64)),
        lt_eq(max(root()), lit(5.0f64))
    )]
    #[case::widening_strengthens_proof(gt(max(masked_input()), lit(5.0f64)), unknown())]
    #[case::zero_nan_count(
        eq(stats::nan_count(masked_input()), lit(0u64)),
        stats::all_non_nan(root())
    )]
    #[case::nonzero_nan_count(eq(stats::nan_count(masked_input()), lit(1u64)), unknown())]
    #[case::null_test(is_null(stats::null_count(masked_input())), lit(false))]
    #[case::fallible_bound(
        lt_eq(
            cast(
                max(masked_input()),
                DType::Primitive(PType::I8, Nullability::Nullable)
            ),
            lit(5i8),
        ),
        unknown()
    )]
    fn uses_only_conservative_facts_from_stored_values(
        #[case] proof: Expression,
        #[case] expected: Expression,
    ) -> VortexResult<()> {
        let dtype = DType::Primitive(PType::F64, Nullability::Nullable);
        assert_eq!(
            weaken_pruning_proof(proof.bind(&dtype)?)?,
            expected.bind(&dtype)?,
        );
        Ok(())
    }
}
