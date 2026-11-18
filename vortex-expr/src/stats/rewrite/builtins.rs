// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::Ordering;

use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions as AggregateEmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::all_nan::AllNan;
use vortex_array::aggregate_fn::fns::all_non_nan::AllNonNan;
use vortex_array::aggregate_fn::fns::all_non_null::AllNonNull;
use vortex_array::aggregate_fn::fns::all_null::AllNull;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::and;
use vortex_array::expr::bound::and_collect;
use vortex_array::expr::bound::cast;
use vortex_array::expr::bound::compare;
use vortex_array::expr::bound::eq;
use vortex_array::expr::bound::gt;
use vortex_array::expr::bound::gt_eq;
use vortex_array::expr::bound::lit;
use vortex_array::expr::bound::lt;
use vortex_array::expr::bound::lt_eq;
use vortex_array::expr::bound::or;
use vortex_array::expr::bound::or_collect;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::StringLike;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::builtin::boolean::Boolean;
use vortex_array::scalar_fn::builtin::cast::Cast;
use vortex_array::scalar_fn::builtin::compare::Compare;
use vortex_array::scalar_fn::builtin::is_not_null::IsNotNull;
use vortex_array::scalar_fn::builtin::is_null::IsNull;
use vortex_array::scalar_fn::builtin::literal::Literal;
use vortex_array::scalar_fn::builtin::operators::BooleanOperator;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_array::scalar_fn::internal::row_count::RowCount;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_functions::between::Between;
use vortex_functions::dynamic::DynamicComparison;
use vortex_functions::dynamic_expression::DynamicExpression;
use vortex_functions::expr::bound::dynamic_with_options;
use vortex_functions::is_nan::IsNan;
use vortex_functions::like::Like;
use vortex_functions::list_contains::ListContains;
use vortex_functions::list_contains::PreparedSet;
use vortex_functions::list_contains::PreparedSetData;
use vortex_functions::lower::Lower;
use vortex_functions::lower_bound::LowerBound;
use vortex_functions::lower_bound::LowerBoundKind;
use vortex_session::VortexSession;

use crate::stats::derive::StatDerivation;
use crate::stats::derive::stat_expr;
use crate::stats::rewrite::StatsRewriteRule;
use crate::stats::rewrite::dynamic_expression::StatsProof;
use crate::stats::rewrite::dynamic_expression::dynamic_stats_proof;
use crate::stats::rewrite::falsify;
use crate::stats::rewrite::like_variant::LikeVariant;
use crate::stats::session::StatsSession;

/// Register built-in stats rewrite rules.
pub(crate) fn register_builtins(session: &StatsSession) {
    session.register_rewrite(CompareNanCountStatsRewrite);
    session.register_rewrite(CompareAllNonNanStatsRewrite);
    session.register_rewrite(BooleanNanCountStatsRewrite);
    session.register_rewrite(BooleanAllNonNanStatsRewrite);
    session.register_rewrite(BetweenStatsRewrite);
    session.register_rewrite(LiteralStatsRewrite);
    session.register_rewrite(IsNullNullCountStatsRewrite);
    session.register_rewrite(IsNullAllNonNullStatsRewrite);
    session.register_rewrite(IsNullAllNullStatsRewrite);
    session.register_rewrite(IsNotNullNullCountStatsRewrite);
    session.register_rewrite(IsNotNullAllNullStatsRewrite);
    session.register_rewrite(IsNotNullAllNonNullStatsRewrite);
    session.register_rewrite(IsNanNaNCountStatsRewrite);
    session.register_rewrite(IsNanAllNonNanStatsRewrite);
    session.register_rewrite(IsNanAllNanStatsRewrite);
    session.register_rewrite(LikeStatsRewrite);
    session.register_rewrite(ListContainsNanCountStatsRewrite);
    session.register_rewrite(ListContainsAllNonNanStatsRewrite);
    session.register_rewrite(DynamicComparisonNanCountStatsRewrite);
    session.register_rewrite(DynamicComparisonAllNonNanStatsRewrite);
    session.register_rewrite(DynamicExpressionStatsRewrite);
    session.register_stat_derivation(LiteralStatDerivation);
    session.register_stat_derivation(CastStatDerivation);
    session.register_stat_derivation(LowerStatDerivation);
}

fn row_count() -> BoundExpression {
    RowCount
        .try_new_bound_expr(EmptyOptions, [])
        .vortex_expect("row-count expressions are always well-typed")
}

#[derive(Debug)]
struct LiteralStatsRewrite;

impl StatsRewriteRule for LiteralStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Literal.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(literal_truth(expr).map(|value| lit(!value)))
    }

    fn satisfy(
        &self,
        expr: &BoundExpression,
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(literal_truth(expr).map(lit))
    }
}

/// A literal's statistics are its value. A statistic that is not meaningful for the literal, such
/// as `NaNCount` over a non-float value, is unknown rather than read from the literal.
#[derive(Debug)]
struct LiteralStatDerivation;

impl StatDerivation for LiteralStatDerivation {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Literal.id()
    }

    fn derive_stat(
        &self,
        expr: &BoundExpression,
        aggregate_fn: &AggregateFnRef,
        _session: &VortexSession,
    ) -> Option<BoundExpression> {
        let scalar = expr.as_::<Literal>();
        if aggregate_fn.is::<Min>() || aggregate_fn.is::<Max>() {
            return Some(lit(scalar.clone()));
        }
        if aggregate_fn.is::<NullCount>() {
            return Some(lit(if scalar.is_null() { 1u64 } else { 0u64 }));
        }
        if aggregate_fn.is::<AllNull>() {
            return Some(lit(scalar.is_null()));
        }
        if aggregate_fn.is::<AllNonNull>() {
            return Some(lit(!scalar.is_null()));
        }
        let value = scalar.as_primitive_opt()?;
        if !value.ptype().is_float() {
            return None;
        }
        if aggregate_fn.is::<NanCount>() {
            return Some(lit(if value.is_nan() { 1u64 } else { 0u64 }));
        }
        // Whether a null is NaN is unknown, so only a non-null value answers these.
        if scalar.is_null() {
            return None;
        }
        if aggregate_fn.is::<AllNan>() {
            return Some(lit(value.is_nan()));
        }
        if aggregate_fn.is::<AllNonNan>() {
            return Some(lit(!value.is_nan()));
        }
        None
    }
}

/// Derives the statistics of `cast(input)` from the statistics of `input`.
#[derive(Debug)]
struct CastStatDerivation;

impl StatDerivation for CastStatDerivation {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Cast.id()
    }

    fn derive_stat(
        &self,
        expr: &BoundExpression,
        aggregate_fn: &AggregateFnRef,
        session: &VortexSession,
    ) -> Option<BoundExpression> {
        let input = expr.child(0);
        if aggregate_fn.is::<Min>() || aggregate_fn.is::<Max>() {
            return stat_expr(input, aggregate_fn, session)
                .map(|stat| cast(stat, expr.as_::<Cast>().clone()));
        }
        // A cast keeps NaNs as NaNs. It changes the sum's dtype and the size of each value, so
        // neither carries over from the input.
        if aggregate_fn.is::<NanCount>() {
            return stat_expr(input, aggregate_fn, session);
        }
        None
    }
}

/// Bounds the statistics of `lower(input)` from the byte-ordered min/max of `input`.
#[derive(Debug)]
struct LowerStatDerivation;

impl StatDerivation for LowerStatDerivation {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Lower.id()
    }

    fn derive_stat(
        &self,
        expr: &BoundExpression,
        aggregate_fn: &AggregateFnRef,
        session: &VortexSession,
    ) -> Option<BoundExpression> {
        let input = expr.child(0);
        let kind = if aggregate_fn.is::<Min>() {
            LowerBoundKind::Min
        } else if aggregate_fn.is::<Max>() {
            LowerBoundKind::Max
        } else if aggregate_fn.is::<NullCount>() {
            // Lowercasing maps nulls to nulls and values to values.
            return null_count(input, session);
        } else {
            return None;
        };
        LowerBound
            .try_new_bound_expr(kind, [min(input, session)?, max(input, session)?])
            .ok()
    }
}

/// Bool literal predicate value
fn literal_truth(expr: &BoundExpression) -> Option<bool> {
    expr.as_::<Literal>().as_bool_opt().and_then(|b| b.value())
}

#[derive(Debug)]
struct CompareNanCountStatsRewrite;

impl StatsRewriteRule for CompareNanCountStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Compare.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        compare_falsify::<NanCountProof>(expr, session)
    }
}

#[derive(Debug)]
struct CompareAllNonNanStatsRewrite;

impl StatsRewriteRule for CompareAllNonNanStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Compare.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        compare_falsify::<AllNonNanProof>(expr, session)
    }
}

#[derive(Debug)]
struct BooleanNanCountStatsRewrite;

impl StatsRewriteRule for BooleanNanCountStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Boolean.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        boolean_falsify::<NanCountProof>(expr, session)
    }
}

#[derive(Debug)]
struct BooleanAllNonNanStatsRewrite;

impl StatsRewriteRule for BooleanAllNonNanStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Boolean.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        boolean_falsify::<AllNonNanProof>(expr, session)
    }
}

fn compare_falsify<P: NonNanProof>(
    expr: &BoundExpression,
    session: &VortexSession,
) -> VortexResult<Option<BoundExpression>> {
    let operator = expr.as_::<Compare>();
    let lhs = expr.child(0);
    let rhs = expr.child(1);

    Ok(match operator {
        CompareOperator::Eq => {
            let left = min(lhs, session)
                .zip(max(rhs, session))
                .map(|(a, b)| gt(a, b));
            let right = min(rhs, session)
                .zip(max(lhs, session))
                .map(|(a, b)| gt(a, b));
            or_collect(left.into_iter().chain(right))
                .map(|value_predicate| {
                    with_non_nan_guards::<P>([lhs, rhs], value_predicate, session)
                })
                .transpose()?
                .flatten()
        }
        CompareOperator::NotEq => min(lhs, session)
            .zip(max(rhs, session))
            .zip(max(lhs, session).zip(min(rhs, session)))
            .map(|((min_lhs, max_rhs), (max_lhs, min_rhs))| {
                with_non_nan_guards::<P>(
                    [lhs, rhs],
                    and(eq(min_lhs, max_rhs), eq(max_lhs, min_rhs)),
                    session,
                )
            })
            .transpose()?
            .flatten(),
        CompareOperator::Gt => max(lhs, session)
            .zip(min(rhs, session))
            .map(|(a, b)| with_non_nan_guards::<P>([lhs, rhs], lt_eq(a, b), session))
            .transpose()?
            .flatten(),
        CompareOperator::Gte => max(lhs, session)
            .zip(min(rhs, session))
            .map(|(a, b)| with_non_nan_guards::<P>([lhs, rhs], lt(a, b), session))
            .transpose()?
            .flatten(),
        CompareOperator::Lt => min(lhs, session)
            .zip(max(rhs, session))
            .map(|(a, b)| with_non_nan_guards::<P>([lhs, rhs], gt_eq(a, b), session))
            .transpose()?
            .flatten(),
        CompareOperator::Lte => min(lhs, session)
            .zip(max(rhs, session))
            .map(|(a, b)| with_non_nan_guards::<P>([lhs, rhs], gt(a, b), session))
            .transpose()?
            .flatten(),
    })
}

fn boolean_falsify<P: NonNanProof>(
    expr: &BoundExpression,
    session: &VortexSession,
) -> VortexResult<Option<BoundExpression>> {
    let operator = expr.as_::<Boolean>();
    let lhs = expr.child(0);
    let rhs = expr.child(1);

    Ok(match operator {
        BooleanOperator::And => {
            if !P::EMIT_UNGUARDED_REWRITES {
                return Ok(None);
            }

            let lhs_falsifier = falsify(lhs, session)?;
            let rhs_falsifier = falsify(rhs, session)?;
            or_collect(lhs_falsifier.into_iter().chain(rhs_falsifier))
        }
        BooleanOperator::Or => {
            // Check before recursing: falsifying the children first would repeat the
            // whole recursive rewrite once per registered `Boolean` rule, which is
            // exponential in `Or`-nesting depth for chains like `a = 1 OR a = 2 OR ...`.
            if !P::EMIT_UNGUARDED_REWRITES {
                return Ok(None);
            }

            match (falsify(lhs, session)?, falsify(rhs, session)?) {
                (Some(lhs), Some(rhs)) => Some(and(lhs, rhs)),
                _ => None,
            }
        }
    })
}

#[derive(Debug)]
struct BetweenStatsRewrite;

impl StatsRewriteRule for BetweenStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Between.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        let options = expr.as_::<Between>();
        let arr = expr.child(0).clone();
        let lower = expr.child(1).clone();
        let upper = expr.child(2).clone();

        let lhs = compare(options.lower_strict.to_operator(), lower, arr.clone());
        let rhs = compare(options.upper_strict.to_operator(), arr, upper);
        falsify(&and(lhs, rhs), session)
    }
}

#[derive(Debug)]
struct IsNullNullCountStatsRewrite;

impl StatsRewriteRule for IsNullNullCountStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        IsNull.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(null_count(expr.child(0), session).map(|null_count| eq(null_count, lit(0u64))))
    }

    fn satisfy(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(null_count(expr.child(0), session).map(|null_count| eq(null_count, row_count())))
    }
}

#[derive(Debug)]
struct IsNullAllNonNullStatsRewrite;

impl StatsRewriteRule for IsNullAllNonNullStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        IsNull.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(all_non_null(expr.child(0), session))
    }
}

#[derive(Debug)]
struct IsNullAllNullStatsRewrite;

impl StatsRewriteRule for IsNullAllNullStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        IsNull.id()
    }

    fn satisfy(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(all_null(expr.child(0), session))
    }
}

#[derive(Debug)]
struct IsNotNullNullCountStatsRewrite;

impl StatsRewriteRule for IsNotNullNullCountStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        IsNotNull.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(null_count(expr.child(0), session).map(|null_count| eq(null_count, row_count())))
    }

    fn satisfy(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(null_count(expr.child(0), session).map(|null_count| eq(null_count, lit(0u64))))
    }
}

#[derive(Debug)]
struct IsNotNullAllNullStatsRewrite;

impl StatsRewriteRule for IsNotNullAllNullStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        IsNotNull.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(all_null(expr.child(0), session))
    }
}

#[derive(Debug)]
struct IsNotNullAllNonNullStatsRewrite;

impl StatsRewriteRule for IsNotNullAllNonNullStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        IsNotNull.id()
    }

    fn satisfy(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(all_non_null(expr.child(0), session))
    }
}

/// Rewrites `is_nan` using the `NaNCount` stat: the predicate is falsified when a zone
/// contains no NaN values, and satisfied when every row in the zone is NaN.
#[derive(Debug)]
struct IsNanNaNCountStatsRewrite;

impl StatsRewriteRule for IsNanNaNCountStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        IsNan.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(nan_count(expr.child(0), session).map(|nan_count| eq(nan_count, lit(0u64))))
    }

    fn satisfy(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(nan_count(expr.child(0), session).map(|nan_count| eq(nan_count, row_count())))
    }
}

/// Falsifies `is_nan` when the `AllNonNan` pruning stat proves that no value in the zone
/// is NaN.
#[derive(Debug)]
struct IsNanAllNonNanStatsRewrite;

impl StatsRewriteRule for IsNanAllNonNanStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        IsNan.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(all_non_nan(expr.child(0), session))
    }
}

/// Satisfies `is_nan` when the `AllNan` pruning stat proves that every value in the zone
/// is NaN.
#[derive(Debug)]
struct IsNanAllNanStatsRewrite;

impl StatsRewriteRule for IsNanAllNanStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        IsNan.id()
    }

    fn satisfy(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(all_nan(expr.child(0), session))
    }
}

#[derive(Debug)]
struct LikeStatsRewrite;

impl StatsRewriteRule for LikeStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Like.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        let like_options = expr.as_::<Like>();
        if like_options.negated || like_options.case_insensitive {
            return Ok(None);
        }

        let Some(pattern) = expr.child(1).as_opt::<Literal>() else {
            return Ok(None);
        };
        let Some(pattern) = pattern.as_utf8().value() else {
            return Ok(None);
        };

        let source = expr.child(0);
        Ok(match LikeVariant::from_str(pattern) {
            Some(LikeVariant::Exact(text)) => {
                min(source, session)
                    .zip(max(source, session))
                    .map(|(source_min, source_max)| {
                        or(
                            gt(source_min, lit(text.as_ref())),
                            lt(source_max, lit(text.as_ref())),
                        )
                    })
            }
            Some(LikeVariant::Prefix(prefix)) => {
                let Some(successor) = prefix.to_string().increment().ok() else {
                    return Ok(None);
                };
                min(source, session)
                    .zip(max(source, session))
                    .map(|(source_min, source_max)| {
                        or(
                            gt_eq(source_min, lit(successor)),
                            lt(source_max, lit(prefix.as_ref())),
                        )
                    })
            }
            None => None,
        })
    }
}

#[derive(Debug)]
struct ListContainsNanCountStatsRewrite;

impl StatsRewriteRule for ListContainsNanCountStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        ListContains.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        list_contains_falsify::<NanCountProof>(expr, session)
    }
}

#[derive(Debug)]
struct ListContainsAllNonNanStatsRewrite;

impl StatsRewriteRule for ListContainsAllNonNanStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        ListContains.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        list_contains_falsify::<AllNonNanProof>(expr, session)
    }
}

/// Lists with more elements falsify `needle IN list` with the bounds of the list rather than
/// with a test per element, which keeps the pruning expression, and its evaluation per zone,
/// small for the large lists of dynamic join filters.
const MAX_LIST_ELEMENT_FALSIFICATIONS: usize = 16;

/// The smallest and largest non-null elements, or `None` when there are none or two elements
/// are unordered.
fn non_null_bounds(elements: &[Scalar]) -> Option<(&Scalar, &Scalar)> {
    let mut non_null = elements.iter().filter(|element| !element.is_null());
    let first = non_null.next()?;
    non_null.try_fold((first, first), |(low, high), element| {
        let low = match element.partial_cmp(low)? {
            Ordering::Less => element,
            Ordering::Equal | Ordering::Greater => low,
        };
        let high = match element.partial_cmp(high)? {
            Ordering::Greater => element,
            Ordering::Equal | Ordering::Less => high,
        };
        Some((low, high))
    })
}

fn list_contains_falsify<P: NonNanProof>(
    expr: &BoundExpression,
    session: &VortexSession,
) -> VortexResult<Option<BoundExpression>> {
    let list = expr.child(0);
    let needle = expr.child(1);

    // Reduction replaces a literal list with a set prepared from it.
    let elements = list
        .as_opt::<Literal>()
        .or_else(|| list.as_opt::<PreparedSet>().map(PreparedSetData::list))
        .and_then(|list| list.as_list_opt())
        .and_then(|list| list.elements());
    let Some(elements) = elements else {
        return Ok(None);
    };
    if elements.is_empty() {
        return Ok(P::EMIT_UNGUARDED_REWRITES.then(|| lit(true)));
    }

    let Some(value_max) = max(needle, session) else {
        return Ok(None);
    };
    let Some(value_min) = min(needle, session) else {
        return Ok(None);
    };

    let value_predicate = if elements.len() > MAX_LIST_ELEMENT_FALSIFICATIONS
        && let Some((low, high)) = non_null_bounds(&elements)
    {
        Some(or(
            lt(value_max, lit(low.clone())),
            gt(value_min, lit(high.clone())),
        ))
    } else {
        and_collect(elements.iter().map(|value| {
            or(
                lt(value_max.clone(), lit(value.clone())),
                gt(value_min.clone(), lit(value.clone())),
            )
        }))
    };
    value_predicate
        .map(|value_predicate| with_non_nan_guards::<P>([needle], value_predicate, session))
        .transpose()
        .map(Option::flatten)
}

#[derive(Debug)]
struct DynamicComparisonNanCountStatsRewrite;

impl StatsRewriteRule for DynamicComparisonNanCountStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        DynamicComparison.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        dynamic_comparison_falsify::<NanCountProof>(expr, session)
    }
}

#[derive(Debug)]
struct DynamicComparisonAllNonNanStatsRewrite;

impl StatsRewriteRule for DynamicComparisonAllNonNanStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        DynamicComparison.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        dynamic_comparison_falsify::<AllNonNanProof>(expr, session)
    }
}

fn dynamic_comparison_falsify<P: NonNanProof>(
    expr: &BoundExpression,
    session: &VortexSession,
) -> VortexResult<Option<BoundExpression>> {
    let dynamic = expr.as_::<DynamicComparison>();
    let lhs = expr.child(0);

    let Some((operator, lhs_stat)) = (match dynamic.operator() {
        CompareOperator::Eq | CompareOperator::NotEq => None,
        CompareOperator::Gt => max(lhs, session).map(|lhs_stat| (CompareOperator::Lte, lhs_stat)),
        CompareOperator::Gte => max(lhs, session).map(|lhs_stat| (CompareOperator::Lt, lhs_stat)),
        CompareOperator::Lt => min(lhs, session).map(|lhs_stat| (CompareOperator::Gte, lhs_stat)),
        CompareOperator::Lte => min(lhs, session).map(|lhs_stat| (CompareOperator::Gt, lhs_stat)),
    }) else {
        return Ok(None);
    };

    let value_predicate = dynamic_with_options(
        dynamic.with_operator(operator, !dynamic.default()),
        lhs_stat,
    );
    with_non_nan_guards::<P>([lhs], value_predicate, session)
}

// Statistics follow NaN-skipping semantics, so min and max request it explicitly.
fn min(expr: &BoundExpression, session: &VortexSession) -> Option<BoundExpression> {
    stat_expr(
        expr,
        &Min.bind(NumericalAggregateOpts::skip_nans()),
        session,
    )
}

fn max(expr: &BoundExpression, session: &VortexSession) -> Option<BoundExpression> {
    stat_expr(
        expr,
        &Max.bind(NumericalAggregateOpts::skip_nans()),
        session,
    )
}

fn null_count(expr: &BoundExpression, session: &VortexSession) -> Option<BoundExpression> {
    stat_expr(expr, &NullCount.bind(AggregateEmptyOptions), session)
}

fn nan_count(expr: &BoundExpression, session: &VortexSession) -> Option<BoundExpression> {
    stat_expr(expr, &NanCount.bind(AggregateEmptyOptions), session)
}

fn all_null(expr: &BoundExpression, session: &VortexSession) -> Option<BoundExpression> {
    stat_expr(expr, &AllNull.bind(AggregateEmptyOptions), session)
}

fn all_non_null(expr: &BoundExpression, session: &VortexSession) -> Option<BoundExpression> {
    stat_expr(expr, &AllNonNull.bind(AggregateEmptyOptions), session)
}

fn all_nan(expr: &BoundExpression, session: &VortexSession) -> Option<BoundExpression> {
    stat_expr(expr, &AllNan.bind(AggregateEmptyOptions), session)
}

fn all_non_nan(expr: &BoundExpression, session: &VortexSession) -> Option<BoundExpression> {
    stat_expr(expr, &AllNonNan.bind(AggregateEmptyOptions), session)
}

enum NanCheck {
    NotNeeded,
    Check(BoundExpression),
    Unavailable,
}

trait NonNanProof {
    const EMIT_UNGUARDED_REWRITES: bool;

    fn check(expr: &BoundExpression, session: &VortexSession) -> VortexResult<NanCheck>;
}

struct NanCountProof;

impl NonNanProof for NanCountProof {
    const EMIT_UNGUARDED_REWRITES: bool = true;

    fn check(expr: &BoundExpression, session: &VortexSession) -> VortexResult<NanCheck> {
        non_nan_check(expr, |expr| match nan_count(expr, session) {
            Some(nan_count) => NanCheck::Check(eq(nan_count, lit(0u64))),
            None => NanCheck::Unavailable,
        })
    }
}

struct AllNonNanProof;

impl NonNanProof for AllNonNanProof {
    const EMIT_UNGUARDED_REWRITES: bool = false;

    fn check(expr: &BoundExpression, session: &VortexSession) -> VortexResult<NanCheck> {
        non_nan_check(expr, |expr| match all_non_nan(expr, session) {
            Some(all_non_nan) => NanCheck::Check(all_non_nan),
            None => NanCheck::Unavailable,
        })
    }
}

// Min/max do not order NaN values, so comparison rewrites are only sound when every
// candidate value is known to be non-NaN. Cast result dtypes are not enough: a cast
// from float to non-float still needs a proof about the float source values.
fn non_nan_check(
    expr: &BoundExpression,
    proof: impl FnOnce(&BoundExpression) -> NanCheck,
) -> VortexResult<NanCheck> {
    if let Some(scalar) = expr.as_opt::<Literal>() {
        if !has_nans(scalar.dtype()) {
            return Ok(NanCheck::NotNeeded);
        }
        let Some(primitive) = scalar.as_primitive_opt() else {
            return Ok(NanCheck::Unavailable);
        };
        return Ok(if primitive.is_nan() {
            NanCheck::Check(lit(false))
        } else {
            NanCheck::NotNeeded
        });
    }

    if expr.is::<Cast>() {
        if !has_nans(expr.child(0).dtype()) {
            return Ok(NanCheck::NotNeeded);
        }

        return non_nan_check(expr.child(0), proof);
    }

    if !has_nans(expr.dtype()) {
        return Ok(NanCheck::NotNeeded);
    }

    Ok(proof(expr))
}

fn has_nans(dtype: &DType) -> bool {
    match dtype {
        DType::Extension(ext) => has_nans(ext.storage_dtype()),
        _ => dtype.is_float(),
    }
}

fn with_non_nan_guards<'a, P: NonNanProof>(
    exprs: impl IntoIterator<Item = &'a BoundExpression>,
    value_predicate: BoundExpression,
    session: &VortexSession,
) -> VortexResult<Option<BoundExpression>> {
    let mut nan_checks = Vec::new();
    for expr in exprs {
        match P::check(expr, session)? {
            NanCheck::NotNeeded => {}
            NanCheck::Check(check) => nan_checks.push(check),
            NanCheck::Unavailable => return Ok(None),
        }
    }
    let nan_predicate = and_collect(nan_checks);

    Ok(match nan_predicate {
        Some(nan_check) => Some(and(nan_check, value_predicate)),
        // No possible NaN-bearing expression remains, so the value predicate is
        // already guarded. Only one registered rule emits this unguarded
        // rewrite so non-float comparisons are not duplicated.
        None if P::EMIT_UNGUARDED_REWRITES => Some(value_predicate),
        None => None,
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use vortex_array::EmptyMetadata;
    use vortex_array::aggregate_fn::AggregateFnRef;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions as AggregateEmptyOptions;
    use vortex_array::aggregate_fn::fns::all_non_nan::AllNonNan;
    use vortex_array::dtype::Claim;
    use vortex_array::dtype::Claims;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::dtype::extension::ExtDType;
    use vortex_array::dtype::extension::ExtId;
    use vortex_array::dtype::extension::ExtVTable;
    use vortex_array::expr::BoundExpression;
    use vortex_array::expr::Expression;
    use vortex_array::expr::and;
    use vortex_array::expr::bound;
    use vortex_array::expr::cast;
    use vortex_array::expr::col;
    use vortex_array::expr::eq;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::gt_eq;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt;
    use vortex_array::expr::lt_eq;
    use vortex_array::expr::or;
    use vortex_array::expr::root;
    use vortex_array::expr::stats::Stat;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar::ScalarValue;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnId;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::scalar_fn::ScalarFnVTableExt;
    use vortex_array::scalar_fn::builtin::boolean::Boolean;
    use vortex_array::scalar_fn::builtin::compare::Compare;
    use vortex_array::scalar_fn::builtin::operators::CompareOperator;
    use vortex_array::scalar_fn::internal::row_count::RowCount;
    use vortex_error::VortexResult;
    use vortex_functions::between::BetweenOptions;
    use vortex_functions::between::StrictComparison;
    use vortex_functions::dynamic::DynamicComparison;
    use vortex_functions::expr::between;
    use vortex_functions::expr::dynamic;
    use vortex_functions::expr::is_nan;
    use vortex_functions::expr::like;
    use vortex_functions::expr::list_contains;
    use vortex_functions::list_contains::ListContains;
    use vortex_functions::list_contains::ListContainsOptions;
    use vortex_functions::list_contains::PreparedSet;
    use vortex_functions::list_contains::PreparedSetData;
    use vortex_session::VortexSession;
    use vortex_session::registry::CachedId;

    use crate::stats::expr::StatFn;
    use crate::stats::expr::StatOptions;
    use crate::stats::rewrite::BoundExpressionPruneExt;
    use crate::stats::rewrite::StatsRewriteRule;
    use crate::stats::session::StatsSessionExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn stat(expr: Expression, stat: Stat) -> Expression {
        let aggregate_fn = stat.aggregate_fn().expect("stat should have aggregate fn");
        stat_fn(expr, aggregate_fn)
    }

    fn stat_fn(expr: Expression, aggregate_fn: AggregateFnRef) -> Expression {
        StatFn.new_expr(StatOptions::new(aggregate_fn), [expr])
    }

    fn test_scope() -> DType {
        DType::Struct(
            StructFields::from_iter([
                ("a", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("b", DType::Primitive(PType::I32, Nullability::NonNullable)),
                ("f", DType::Primitive(PType::F32, Nullability::NonNullable)),
                ("s", DType::Utf8(Nullability::NonNullable)),
                ("t", DType::Utf8(Nullability::NonNullable)),
                ("n", nested_struct_dtype()),
            ]),
            Nullability::NonNullable,
        )
    }

    fn nested_struct_dtype() -> DType {
        DType::Struct(
            StructFields::from_iter([("x", DType::Primitive(PType::F32, Nullability::Nullable))]),
            Nullability::NonNullable,
        )
    }

    fn falsify(expr: &Expression) -> VortexResult<Option<BoundExpression>> {
        expr.bind(&test_scope())?.falsify(&SESSION)
    }

    fn satisfy(expr: &Expression) -> VortexResult<Option<BoundExpression>> {
        expr.bind(&test_scope())?.satisfy(&SESSION)
    }

    fn bind_expected(expr: Option<Expression>) -> VortexResult<Option<BoundExpression>> {
        expr.map(|expr| expr.bind(&test_scope())).transpose()
    }

    fn all_null(expr: &Expression) -> Expression {
        crate::stats::all_null(expr.clone())
    }

    fn all_non_null(expr: &Expression) -> Expression {
        crate::stats::all_non_null(expr.clone())
    }

    fn all_nan(expr: &Expression) -> Expression {
        crate::stats::all_nan(expr.clone())
    }

    fn all_non_nan(expr: &Expression) -> Expression {
        crate::stats::all_non_nan(expr.clone())
    }

    macro_rules! assert_rewrite_eq {
        ($actual:expr, $expected:expr) => {
            assert_eq!($actual, bind_expected($expected)?)
        };
    }

    fn nan_guarded(expr: Expression, value_predicate: Expression) -> Expression {
        or(
            and(
                eq(stat(expr.clone(), Stat::NaNCount), lit(0u64)),
                value_predicate.clone(),
            ),
            and(
                stat_fn(expr, AllNonNan.bind(AggregateEmptyOptions)),
                value_predicate,
            ),
        )
    }

    #[test]
    fn pruning_skips_unclaimed_literal_bounds() {
        let literal = bound::lit(Scalar::null(DType::Variant(Nullability::Nullable)));

        assert!(super::min(&literal, &SESSION).is_none());
        assert!(super::max(&literal, &SESSION).is_none());
        assert!(super::null_count(&literal, &SESSION).is_some());
    }

    #[derive(Clone, Debug, PartialEq, Eq, Hash)]
    struct OrderedFloat;

    impl ExtVTable for OrderedFloat {
        type Metadata = EmptyMetadata;
        type StoredMetadata = EmptyMetadata;
        type NativeValue<'a> = &'a ScalarValue;

        fn id(&self) -> ExtId {
            static ID: CachedId = CachedId::new("test.ordered_float");
            *ID
        }

        fn serialize_metadata(&self, _metadata: &EmptyMetadata) -> VortexResult<Vec<u8>> {
            Ok(vec![])
        }

        fn deserialize_metadata(&self, _bytes: &[u8]) -> VortexResult<EmptyMetadata> {
            Ok(EmptyMetadata)
        }

        fn metadata_from_storage(
            &self,
            metadata: EmptyMetadata,
            _storage_dtype: &DType,
        ) -> VortexResult<EmptyMetadata> {
            Ok(metadata)
        }

        fn storage_dtype(_metadata: &EmptyMetadata) -> DType {
            PType::F64.into()
        }

        fn claims(_metadata: &EmptyMetadata) -> Claims {
            Claims::none().with(Claim::Ord)
        }

        fn unpack_native<'a>(
            _ext_dtype: &'a ExtDType<Self>,
            storage_value: &'a ScalarValue,
        ) -> VortexResult<Self::NativeValue<'a>> {
            Ok(storage_value)
        }
    }

    #[test]
    fn bounds_skip_extensions_stored_as_floats() -> VortexResult<()> {
        let ext_dtype =
            ExtDType::try_with_vtable(OrderedFloat, EmptyMetadata, Nullability::NonNullable)?
                .erased();
        let literal = Scalar::try_extension_ref(ext_dtype.clone(), Scalar::from(5.0f64))?;
        let dtype = DType::Extension(ext_dtype);
        let predicate = gt(root(), lit(literal)).bind(&dtype)?;
        assert!(predicate.falsify(&SESSION)?.is_none());

        let dtype = PType::F64.into();
        let predicate = gt(root(), lit(5.0f64)).bind(&dtype)?;
        let guarded = nan_guarded(root(), lt_eq(stat(root(), Stat::Max), lit(5.0f64)));
        assert_eq!(predicate.falsify(&SESSION)?, Some(guarded.bind(&dtype)?));
        Ok(())
    }

    #[test]
    fn rewrites_comparison_falsifier() -> VortexResult<()> {
        let expr = gt(col("a"), lit(10));
        assert_rewrite_eq!(
            falsify(&expr)?,
            Some(lt_eq(stat(col("a"), Stat::Max), lit(10)))
        );

        let expr = eq(col("a"), col("b"));
        assert_rewrite_eq!(
            falsify(&expr)?,
            Some(or(
                gt(stat(col("a"), Stat::Min), stat(col("b"), Stat::Max)),
                gt(stat(col("b"), Stat::Min), stat(col("a"), Stat::Max)),
            ))
        );

        let expr = eq(col("s"), col("t"));
        assert_rewrite_eq!(
            falsify(&expr)?,
            Some(or(
                gt(stat(col("s"), Stat::Min), stat(col("t"), Stat::Max)),
                gt(stat(col("t"), Stat::Min), stat(col("s"), Stat::Max)),
            ))
        );
        Ok(())
    }

    #[test]
    fn rewrites_boolean_falsifiers() -> VortexResult<()> {
        let expr = and(gt(col("a"), lit(10)), lt(col("a"), lit(50)));
        assert_rewrite_eq!(
            falsify(&expr)?,
            Some(or(
                lt_eq(stat(col("a"), Stat::Max), lit(10)),
                gt_eq(stat(col("a"), Stat::Min), lit(50)),
            ))
        );

        let expr = or(gt(col("a"), lit(10)), lt(col("a"), lit(5)));
        assert_rewrite_eq!(
            falsify(&expr)?,
            Some(and(
                lt_eq(stat(col("a"), Stat::Max), lit(10)),
                gt_eq(stat(col("a"), Stat::Min), lit(5)),
            ))
        );
        Ok(())
    }

    /// Counts how many times the stats rewrite visits a node of one scalar function.
    #[derive(Debug)]
    struct VisitCounter(ScalarFnId, Arc<AtomicUsize>);

    impl StatsRewriteRule for VisitCounter {
        fn scalar_fn_id(&self) -> ScalarFnId {
            self.0
        }

        fn falsify(
            &self,
            _expr: &BoundExpression,
            _session: &VortexSession,
        ) -> VortexResult<Option<BoundExpression>> {
            self.1.fetch_add(1, Ordering::Relaxed);
            Ok(None)
        }
    }

    #[test]
    fn or_chain_falsify_visits_each_node_once() -> VortexResult<()> {
        let session = VortexSession::empty();
        let visits = Arc::new(AtomicUsize::new(0));
        session
            .stats()
            .register_rewrite(VisitCounter(Compare.id(), Arc::clone(&visits)));
        session
            .stats()
            .register_rewrite(VisitCounter(Boolean.id(), Arc::clone(&visits)));

        let expr = (1..16).fold(eq(col("a"), lit(0)), |chain, i| {
            or(chain, eq(col("a"), lit(i)))
        });
        let falsifier = expr.bind(&test_scope())?.falsify(&session)?;
        assert!(falsifier.is_some());

        // One visit per `Compare` or `Boolean` node: 16 comparisons plus 15 `or`s.
        assert_eq!(visits.load(Ordering::Relaxed), 31);
        Ok(())
    }

    #[test]
    fn rewrites_between_falsifier() -> VortexResult<()> {
        let expr = between(
            col("a"),
            lit(10),
            lit(50),
            BetweenOptions {
                lower_strict: StrictComparison::NonStrict,
                upper_strict: StrictComparison::NonStrict,
            },
        );

        assert_rewrite_eq!(
            falsify(&expr)?,
            Some(or(
                gt(lit(10), stat(col("a"), Stat::Max)),
                gt(stat(col("a"), Stat::Min), lit(50)),
            ))
        );
        Ok(())
    }

    #[test]
    fn rewrites_null_falsifiers() -> VortexResult<()> {
        assert_rewrite_eq!(
            falsify(&is_null(col("a")))?,
            Some(or(
                eq(stat(col("a"), Stat::NullCount), lit(0u64)),
                all_non_null(&col("a")),
            ))
        );

        assert_rewrite_eq!(
            falsify(&is_not_null(col("a")))?,
            Some(or(
                eq(
                    stat(col("a"), Stat::NullCount),
                    RowCount.new_expr(EmptyOptions, []),
                ),
                all_null(&col("a")),
            ))
        );
        Ok(())
    }

    #[test]
    fn rewrites_null_satisfiers() -> VortexResult<()> {
        assert_rewrite_eq!(
            satisfy(&is_null(col("a")))?,
            Some(or(
                eq(
                    stat(col("a"), Stat::NullCount),
                    RowCount.new_expr(EmptyOptions, []),
                ),
                all_null(&col("a")),
            ))
        );

        assert_rewrite_eq!(
            satisfy(&is_not_null(col("a")))?,
            Some(or(
                eq(stat(col("a"), Stat::NullCount), lit(0u64)),
                all_non_null(&col("a")),
            ))
        );
        Ok(())
    }

    #[test]
    fn rewrites_nan_falsifiers() -> VortexResult<()> {
        assert_rewrite_eq!(
            falsify(&is_nan(col("f")))?,
            Some(or(
                eq(stat(col("f"), Stat::NaNCount), lit(0u64)),
                all_non_nan(&col("f")),
            ))
        );

        // Nullable floats prune on nan_count just the same.
        assert_rewrite_eq!(
            falsify(&is_nan(get_item("x", col("n"))))?,
            Some(or(
                eq(stat(get_item("x", col("n")), Stat::NaNCount), lit(0u64)),
                all_non_nan(&get_item("x", col("n"))),
            ))
        );

        // NaN literals fold through the nan_count and all_non_nan stats.
        assert_rewrite_eq!(
            falsify(&is_nan(lit(f32::NAN)))?,
            Some(or(eq(lit(1u64), lit(0u64)), lit(false)))
        );
        Ok(())
    }

    #[test]
    fn rewrites_nan_satisfiers() -> VortexResult<()> {
        assert_rewrite_eq!(
            satisfy(&is_nan(col("f")))?,
            Some(or(
                eq(
                    stat(col("f"), Stat::NaNCount),
                    RowCount.new_expr(EmptyOptions, [])
                ),
                all_nan(&col("f")),
            ))
        );

        assert_rewrite_eq!(
            satisfy(&is_nan(lit(f32::NAN)))?,
            Some(or(
                eq(lit(1u64), RowCount.new_expr(EmptyOptions, [])),
                lit(true),
            ))
        );
        Ok(())
    }

    #[test]
    fn rewrites_list_contains_falsifier() -> VortexResult<()> {
        let list = Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![1.into(), 2.into(), 3.into()],
            Nullability::NonNullable,
        );
        let expr = list_contains(lit(list), col("a"));

        assert_rewrite_eq!(
            falsify(&expr)?,
            Some(and(
                and(
                    or(
                        lt(stat(col("a"), Stat::Max), lit(1i32)),
                        gt(stat(col("a"), Stat::Min), lit(1i32)),
                    ),
                    or(
                        lt(stat(col("a"), Stat::Max), lit(2i32)),
                        gt(stat(col("a"), Stat::Min), lit(2i32)),
                    ),
                ),
                or(
                    lt(stat(col("a"), Stat::Max), lit(3i32)),
                    gt(stat(col("a"), Stat::Min), lit(3i32)),
                ),
            ))
        );
        Ok(())
    }

    #[test]
    fn rewrites_large_list_contains_falsifier_to_bounds() -> VortexResult<()> {
        let list = Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::Nullable)),
            (0..20)
                .rev()
                .map(|value| Scalar::from(Some(value)))
                .chain([Scalar::null(DType::Primitive(
                    PType::I32,
                    Nullability::Nullable,
                ))])
                .collect(),
            Nullability::NonNullable,
        );
        let expr = list_contains(lit(list), col("a"));

        assert_rewrite_eq!(
            falsify(&expr)?,
            Some(or(
                lt(stat(col("a"), Stat::Max), lit(Scalar::from(Some(0i32)))),
                gt(stat(col("a"), Stat::Min), lit(Scalar::from(Some(19i32)))),
            ))
        );
        Ok(())
    }

    #[test]
    fn rewrites_prepared_set_list_contains_falsifier() -> VortexResult<()> {
        let list = Scalar::list(
            Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
            vec![1.into(), 2.into()],
            Nullability::NonNullable,
        );
        let set = PreparedSet.new_expr(PreparedSetData::try_new(list)?, Vec::<Expression>::new());
        let expr = ListContains.new_expr(ListContainsOptions::default(), [set, col("a")]);

        assert_rewrite_eq!(
            falsify(&expr)?,
            Some(and(
                or(
                    lt(stat(col("a"), Stat::Max), lit(1i32)),
                    gt(stat(col("a"), Stat::Min), lit(1i32)),
                ),
                or(
                    lt(stat(col("a"), Stat::Max), lit(2i32)),
                    gt(stat(col("a"), Stat::Min), lit(2i32)),
                ),
            ))
        );
        Ok(())
    }

    #[test]
    fn rewrites_like_falsifier() -> VortexResult<()> {
        let expr = like(col("s"), lit("prefix%"));
        assert_rewrite_eq!(
            falsify(&expr)?,
            Some(or(
                gt_eq(stat(col("s"), Stat::Min), lit("prefiy")),
                lt(stat(col("s"), Stat::Max), lit("prefix")),
            ))
        );

        let expr = like(col("s"), lit(r"\%%"));
        assert_rewrite_eq!(
            falsify(&expr)?,
            Some(or(
                gt_eq(stat(col("s"), Stat::Min), lit("&")),
                lt(stat(col("s"), Stat::Max), lit("%")),
            ))
        );

        let expr = like(col("s"), lit("pref%ix%"));
        assert_rewrite_eq!(
            falsify(&expr)?,
            Some(or(
                gt_eq(stat(col("s"), Stat::Min), lit("preg")),
                lt(stat(col("s"), Stat::Max), lit("pref")),
            ))
        );

        let expr = like(col("s"), lit("pref_ix_"));
        assert_rewrite_eq!(
            falsify(&expr)?,
            Some(or(
                gt_eq(stat(col("s"), Stat::Min), lit("preg")),
                lt(stat(col("s"), Stat::Max), lit("pref")),
            ))
        );

        let expr = like(col("s"), lit("exact"));
        assert_rewrite_eq!(
            falsify(&expr)?,
            Some(or(
                gt(stat(col("s"), Stat::Min), lit("exact")),
                lt(stat(col("s"), Stat::Max), lit("exact")),
            ))
        );

        let expr = like(col("s"), lit("%suffix"));
        assert_rewrite_eq!(falsify(&expr)?, None);
        Ok(())
    }

    #[test]
    fn rewrites_dynamic_comparison_falsifier() -> VortexResult<()> {
        let expr = dynamic(
            CompareOperator::Gt,
            || Some(10i32.into()),
            DType::Primitive(PType::I32, Nullability::NonNullable),
            true,
            col("a"),
        );
        let dynamic = expr.as_::<DynamicComparison>();

        assert_rewrite_eq!(
            falsify(&expr)?,
            Some(DynamicComparison.new_expr(
                dynamic.with_operator(CompareOperator::Lte, false),
                [stat(col("a"), Stat::Max)],
            ))
        );
        Ok(())
    }

    #[test]
    fn nan_guard_tracks_cast_source_dtype() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let expr = gt(cast(col("f"), dtype.clone()), lit(5i32));

        assert_rewrite_eq!(
            falsify(&expr)?,
            Some(nan_guarded(
                col("f"),
                lt_eq(cast(stat(col("f"), Stat::Max), dtype), lit(5i32)),
            ))
        );
        Ok(())
    }

    #[test]
    fn skips_falsifier_when_min_max_unsupported_for_dtype() -> VortexResult<()> {
        // Struct comparisons are valid predicates, but min/max aggregates do not
        // support struct inputs, so no stats-backed falsifier should be produced.
        let struct_scalar = Scalar::struct_(
            nested_struct_dtype(),
            vec![Scalar::primitive(1.0f32, Nullability::Nullable)],
        );
        assert_rewrite_eq!(falsify(&lt_eq(col("n"), lit(struct_scalar.clone())))?, None);
        assert_rewrite_eq!(falsify(&eq(col("n"), lit(struct_scalar)))?, None);
        Ok(())
    }

    #[test]
    fn forwards_min_max_through_safe_cast() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let expr = eq(cast(col("a"), dtype.clone()), lit(42i64));

        assert_rewrite_eq!(
            falsify(&expr)?,
            Some(or(
                gt(cast(stat(col("a"), Stat::Min), dtype.clone()), lit(42i64)),
                gt(lit(42i64), cast(stat(col("a"), Stat::Max), dtype)),
            ))
        );
        Ok(())
    }
}

#[derive(Debug)]
struct DynamicExpressionStatsRewrite;

impl StatsRewriteRule for DynamicExpressionStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        DynamicExpression.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        dynamic_stats_proof(expr, session, StatsProof::Falsify)
    }

    fn satisfy(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        dynamic_stats_proof(expr, session, StatsProof::Satisfy)
    }
}
