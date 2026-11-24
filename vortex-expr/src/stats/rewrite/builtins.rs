// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::aggregate_fn::AggregateFnRef;
use std::cmp::Ordering;

use vortex_aggregates::all_nan::AllNan;
use vortex_aggregates::all_non_nan::AllNonNan;
use vortex_aggregates::all_non_null::AllNonNull;
use vortex_aggregates::all_null::AllNull;
use vortex_aggregates::max::Max;
use vortex_aggregates::min::Min;
use vortex_aggregates::nan_count::NanCount;
use vortex_aggregates::null_count::NullCount;

use vortex_aggregates::stat_aggregate_fn;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions as AggregateEmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::builtin::boolean::Boolean;
use vortex_array::scalar_fn::builtin::cast::Cast;
use vortex_array::scalar_fn::builtin::compare::Compare;
use vortex_array::scalar_fn::builtin::is_not_null::IsNotNull;
use vortex_array::scalar_fn::builtin::literal::Literal;
use vortex_array::scalar_fn::builtin::operators::BooleanOperator;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::bound::and;
use crate::bound::and_collect;
use crate::bound::cast;
use crate::bound::eq;
use crate::bound::gt;
use crate::bound::gt_eq;
use crate::bound::lit;
use crate::bound::lt;
use crate::bound::lt_eq;
use crate::bound::or_collect;
use crate::stats::derive::StatDerivation;
use crate::stats::derive::stat_expr;
use crate::stats::rewrite::StatsRewriteRule;
use crate::stats::rewrite::falsify;
use crate::stats::rewrite::helpers::*;
use crate::stats::rewrite::satisfy;
use crate::stats::session::StatsSession;

/// Register the stats rewrite rules for the scalar functions vortex-array is built on.
pub(crate) fn register_builtins(session: &StatsSession) {
    session.register_rewrite(CompareNanCountStatsRewrite);
    session.register_rewrite(CompareAllNonNanStatsRewrite);
    session.register_rewrite(BooleanNanCountStatsRewrite);
    session.register_rewrite(BooleanAllNonNanStatsRewrite);
    session.register_rewrite(LiteralStatsRewrite);
    session.register_rewrite(IsNotNullNullCountStatsRewrite);
    session.register_rewrite(IsNotNullAllNullStatsRewrite);
    session.register_rewrite(IsNotNullAllNonNullStatsRewrite);
    session.register_stat_derivation(LiteralStatDerivation);
    session.register_stat_derivation(CastStatDerivation);
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
        // A cast keeps NaNs as NaNs and nulls as nulls. A cast to a non-nullable dtype fails on a
        // null input, so it does not change the null count either. A cast changes the sum's dtype
        // and the size of each value, so neither carries over from the input.
        if aggregate_fn.is::<NanCount>() || aggregate_fn.is::<NullCount>() {
            return stat_expr(input, aggregate_fn, session);
        }
        None
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

    fn satisfy(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        compare_satisfy(expr, session)
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

    fn satisfy(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        boolean_satisfy(expr, session)
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

/// Proves a comparison true for every row: the bounds of the operands order every pair of
/// values, and neither operand holds a null or a NaN, for which a comparison is not true.
fn compare_satisfy(
    expr: &BoundExpression,
    session: &VortexSession,
) -> VortexResult<Option<BoundExpression>> {
    let operator = expr.as_::<Compare>();
    let lhs = expr.child(0);
    let rhs = expr.child(1);

    let value_predicate = match operator {
        CompareOperator::Gt => min(lhs, session)
            .zip(max(rhs, session))
            .map(|(a, b)| gt(a, b)),
        CompareOperator::Gte => min(lhs, session)
            .zip(max(rhs, session))
            .map(|(a, b)| gt_eq(a, b)),
        CompareOperator::Lt => max(lhs, session)
            .zip(min(rhs, session))
            .map(|(a, b)| lt(a, b)),
        CompareOperator::Lte => max(lhs, session)
            .zip(min(rhs, session))
            .map(|(a, b)| lt_eq(a, b)),
        CompareOperator::Eq | CompareOperator::NotEq => None,
    };
    let Some(value_predicate) = value_predicate else {
        return Ok(None);
    };
    let Some(null_checks) = [lhs, rhs]
        .into_iter()
        .map(|operand| null_count(operand, session).map(|count| eq(count, lit(0u64))))
        .collect::<Option<Vec<_>>>()
    else {
        return Ok(None);
    };
    let guarded = and_collect(null_checks.into_iter().chain([value_predicate]))
        .vortex_expect("at least one predicate");
    with_non_nan_guards::<NanCountProof>([lhs, rhs], guarded, session)
}

fn boolean_satisfy(
    expr: &BoundExpression,
    session: &VortexSession,
) -> VortexResult<Option<BoundExpression>> {
    let operator = expr.as_::<Boolean>();
    let lhs = expr.child(0);
    let rhs = expr.child(1);

    Ok(match operator {
        BooleanOperator::And => match (satisfy(lhs, session)?, satisfy(rhs, session)?) {
            (Some(lhs), Some(rhs)) => Some(and(lhs, rhs)),
            _ => None,
        },
        BooleanOperator::Or => or_collect(
            satisfy(lhs, session)?
                .into_iter()
                .chain(satisfy(rhs, session)?),
        ),
    })
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

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use vortex_aggregates::all_non_nan::AllNonNan;
    use vortex_aggregates::stat_aggregate_fn;
    use vortex_array::aggregate_fn::AggregateFnRef;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions as AggregateEmptyOptions;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnId;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::scalar_fn::builtin::boolean::Boolean;
    use vortex_array::scalar_fn::builtin::compare::Compare;
    use vortex_array::stats::Stat;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::BoundExpression;
    use crate::Expression;
    use crate::ScalarFnExprExt;
    use crate::and;
    use crate::cast;
    use crate::col;
    use crate::eq;
    use crate::gt;
    use crate::gt_eq;
    use crate::is_not_null;
    use crate::lambda::row_count::RowCount;
    use crate::lit;
    use crate::lt;
    use crate::lt_eq;
    use crate::not_eq;
    use crate::or;
    use crate::stats::expr::StatFn;
    use crate::stats::expr::StatOptions;
    use crate::stats::rewrite::BoundExpressionPruneExt;
    use crate::stats::rewrite::StatsRewriteRule;
    use crate::stats::session::StatsSessionExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    fn stat(expr: Expression, stat: Stat) -> Expression {
        let aggregate_fn = stat_aggregate_fn(stat).expect("stat should have aggregate fn");
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

    fn cmp_satisfier(
        lhs_nulls: Expression,
        rhs_nulls: Expression,
        value_predicate: Expression,
    ) -> Expression {
        and(
            and(eq(lhs_nulls, lit(0u64)), eq(rhs_nulls, lit(0u64))),
            value_predicate,
        )
    }

    #[test]
    fn rewrites_comparison_satisfier() -> VortexResult<()> {
        assert_rewrite_eq!(
            satisfy(&gt(col("a"), lit(10)))?,
            Some(cmp_satisfier(
                stat(col("a"), Stat::NullCount),
                lit(0u64),
                gt(stat(col("a"), Stat::Min), lit(10)),
            ))
        );
        assert_rewrite_eq!(
            satisfy(&gt_eq(col("a"), lit(10)))?,
            Some(cmp_satisfier(
                stat(col("a"), Stat::NullCount),
                lit(0u64),
                gt_eq(stat(col("a"), Stat::Min), lit(10)),
            ))
        );
        assert_rewrite_eq!(
            satisfy(&lt(col("a"), lit(10)))?,
            Some(cmp_satisfier(
                stat(col("a"), Stat::NullCount),
                lit(0u64),
                lt(stat(col("a"), Stat::Max), lit(10)),
            ))
        );
        assert_rewrite_eq!(
            satisfy(&lt_eq(col("a"), lit(10)))?,
            Some(cmp_satisfier(
                stat(col("a"), Stat::NullCount),
                lit(0u64),
                lt_eq(stat(col("a"), Stat::Max), lit(10)),
            ))
        );
        assert_rewrite_eq!(satisfy(&eq(col("a"), lit(10)))?, None);
        assert_rewrite_eq!(satisfy(&not_eq(col("a"), lit(10)))?, None);

        // Float comparisons also require a NaNCount == 0 guard.
        assert_rewrite_eq!(
            satisfy(&gt(col("f"), lit(1.0f32)))?,
            Some(and(
                eq(stat(col("f"), Stat::NaNCount), lit(0u64)),
                cmp_satisfier(
                    stat(col("f"), Stat::NullCount),
                    lit(0u64),
                    gt(stat(col("f"), Stat::Min), lit(1.0f32)),
                ),
            ))
        );
        Ok(())
    }

    #[test]
    fn rewrites_boolean_satisfiers() -> VortexResult<()> {
        let lhs_sat = cmp_satisfier(
            stat(col("a"), Stat::NullCount),
            lit(0u64),
            gt(stat(col("a"), Stat::Min), lit(10)),
        );
        let rhs_sat = cmp_satisfier(
            stat(col("a"), Stat::NullCount),
            lit(0u64),
            lt(stat(col("a"), Stat::Max), lit(50)),
        );

        assert_rewrite_eq!(
            satisfy(&and(gt(col("a"), lit(10)), lt(col("a"), lit(50))))?,
            Some(and(lhs_sat.clone(), rhs_sat.clone()))
        );
        assert_rewrite_eq!(
            satisfy(&and(gt(col("a"), lit(10)), eq(col("a"), lit(20))))?,
            None
        );
        assert_rewrite_eq!(
            satisfy(&or(gt(col("a"), lit(10)), lt(col("a"), lit(50))))?,
            Some(or(lhs_sat.clone(), rhs_sat))
        );
        assert_rewrite_eq!(
            satisfy(&or(gt(col("a"), lit(10)), eq(col("a"), lit(20))))?,
            Some(lhs_sat)
        );
        Ok(())
    }

    #[test]
    fn rewrites_null_falsifiers() -> VortexResult<()> {
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
            satisfy(&is_not_null(col("a")))?,
            Some(or(
                eq(stat(col("a"), Stat::NullCount), lit(0u64)),
                all_non_null(&col("a")),
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

    #[test]
    fn satisfies_comparison_through_cast() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let expr = gt(cast(col("a"), dtype.clone()), lit(10i64));

        assert_rewrite_eq!(
            satisfy(&expr)?,
            Some(cmp_satisfier(
                stat(col("a"), Stat::NullCount),
                lit(0u64),
                gt(cast(stat(col("a"), Stat::Min), dtype), lit(10i64)),
            ))
        );
        Ok(())
    }
}
