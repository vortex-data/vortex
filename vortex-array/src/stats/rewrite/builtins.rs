// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::num::NonZeroUsize;

use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions as AggregateEmptyOptions;
use crate::aggregate_fn::fns::all_non_null::AllNonNull;
use crate::aggregate_fn::fns::all_null::AllNull;
use crate::aggregate_fn::fns::bounded_max::BoundedMax;
use crate::aggregate_fn::fns::bounded_max::BoundedMaxOptions;
use crate::aggregate_fn::fns::bounded_min::BoundedMin;
use crate::aggregate_fn::fns::bounded_min::BoundedMinOptions;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::and;
use crate::expr::and_collect;
use crate::expr::cast;
use crate::expr::eq;
use crate::expr::gt;
use crate::expr::gt_eq;
use crate::expr::lit;
use crate::expr::lt;
use crate::expr::lt_eq;
use crate::expr::or;
use crate::expr::or_collect;
use crate::expr::stats::Stat;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::between::Between;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::is_not_null::IsNotNull;
use crate::scalar_fn::fns::is_null::IsNull;
use crate::scalar_fn::fns::list_contains::ListContains;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::operators::Operator;
use crate::scalar_fn::internal::row_count::RowCount;
use crate::stats::expr::StatFn;
use crate::stats::expr::StatOptions;
use crate::stats::rewrite::StatsRewriteCtx;
use crate::stats::rewrite::StatsRewriteRule;
use crate::stats::session::StatsSession;

const DEFAULT_BOUNDED_STAT_MAX_BYTES: usize = 64;

fn default_bounded_stat_max_bytes() -> NonZeroUsize {
    NonZeroUsize::new(DEFAULT_BOUNDED_STAT_MAX_BYTES)
        .vortex_expect("default bounded stat max bytes is non-zero")
}

/// Register built-in stats rewrite rules.
pub(crate) fn register_builtins(session: &StatsSession) {
    session.register_rewrite(BinaryStatsRewrite);
    session.register_rewrite(BoundedBinaryStatsRewrite);
    session.register_rewrite(BetweenStatsRewrite);
    session.register_rewrite(IsNullLegacyStatsRewrite);
    session.register_rewrite(IsNullAllNonNullStatsRewrite);
    session.register_rewrite(IsNullAllNullStatsRewrite);
    session.register_rewrite(IsNotNullLegacyStatsRewrite);
    session.register_rewrite(IsNotNullAllNullStatsRewrite);
    session.register_rewrite(IsNotNullAllNonNullStatsRewrite);
    session.register_rewrite(ListContainsStatsRewrite);
}

#[derive(Debug)]
struct BinaryStatsRewrite;

impl StatsRewriteRule for BinaryStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Binary.id()
    }

    fn falsify(
        &self,
        expr: &Expression,
        ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        let operator = expr.as_::<Binary>();
        let lhs = expr.child(0);
        let rhs = expr.child(1);

        Ok(match operator {
            Operator::Eq => {
                let left = min(lhs).zip(max(rhs)).map(|(a, b)| gt(a, b));
                let right = min(rhs).zip(max(lhs)).map(|(a, b)| gt(a, b));
                or_collect(left.into_iter().chain(right))
            }
            Operator::NotEq => min(lhs).zip(max(rhs)).zip(max(lhs).zip(min(rhs))).map(
                |((min_lhs, max_rhs), (max_lhs, min_rhs))| {
                    and(eq(min_lhs, max_rhs), eq(max_lhs, min_rhs))
                },
            ),
            Operator::Gt => max(lhs).zip(min(rhs)).map(|(a, b)| lt_eq(a, b)),
            Operator::Gte => max(lhs).zip(min(rhs)).map(|(a, b)| lt(a, b)),
            Operator::Lt => min(lhs).zip(max(rhs)).map(|(a, b)| gt_eq(a, b)),
            Operator::Lte => min(lhs).zip(max(rhs)).map(|(a, b)| gt(a, b)),
            Operator::And => {
                let lhs_falsifier = ctx.falsify(lhs)?;
                let rhs_falsifier = ctx.falsify(rhs)?;
                or_collect(lhs_falsifier.into_iter().chain(rhs_falsifier))
            }
            Operator::Or => match (ctx.falsify(lhs)?, ctx.falsify(rhs)?) {
                (Some(lhs), Some(rhs)) => Some(and(lhs, rhs)),
                _ => None,
            },
            Operator::Add | Operator::Sub | Operator::Mul | Operator::Div => None,
        })
    }
}

#[derive(Debug)]
struct BoundedBinaryStatsRewrite;

impl StatsRewriteRule for BoundedBinaryStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Binary.id()
    }

    fn falsify(
        &self,
        expr: &Expression,
        _ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        let operator = expr.as_::<Binary>();
        let lhs = expr.child(0);
        let rhs = expr.child(1);

        Ok(match operator {
            Operator::Eq => {
                let left = gt(bounded_min(lhs), bounded_max(rhs));
                let right = gt(bounded_min(rhs), bounded_max(lhs));
                Some(or(left, right))
            }
            Operator::Gt => Some(lt_eq(bounded_max(lhs), bounded_min(rhs))),
            Operator::Gte => Some(lt(bounded_max(lhs), bounded_min(rhs))),
            Operator::Lt => Some(gt_eq(bounded_min(lhs), bounded_max(rhs))),
            Operator::Lte => Some(gt(bounded_min(lhs), bounded_max(rhs))),
            Operator::NotEq
            | Operator::And
            | Operator::Or
            | Operator::Add
            | Operator::Sub
            | Operator::Mul
            | Operator::Div => None,
        })
    }
}

#[derive(Debug)]
struct BetweenStatsRewrite;

impl StatsRewriteRule for BetweenStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Between.id()
    }

    fn falsify(
        &self,
        expr: &Expression,
        ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        let options = expr.as_::<Between>();
        let arr = expr.child(0).clone();
        let lower = expr.child(1).clone();
        let upper = expr.child(2).clone();

        let lhs = Binary.new_expr(options.lower_strict.to_operator(), [lower, arr.clone()]);
        let rhs = Binary.new_expr(options.upper_strict.to_operator(), [arr, upper]);
        ctx.falsify(&and(lhs, rhs))
    }
}

#[derive(Debug)]
struct IsNullLegacyStatsRewrite;

impl StatsRewriteRule for IsNullLegacyStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        IsNull.id()
    }

    fn falsify(
        &self,
        expr: &Expression,
        _ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        Ok(null_count(expr.child(0)).map(|null_count| eq(null_count, lit(0u64))))
    }

    fn satisfy(
        &self,
        expr: &Expression,
        _ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        Ok(null_count(expr.child(0))
            .map(|null_count| eq(null_count, RowCount.new_expr(EmptyOptions, []))))
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
        expr: &Expression,
        _ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(all_non_null(expr.child(0))))
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
        expr: &Expression,
        _ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(all_null(expr.child(0))))
    }
}

#[derive(Debug)]
struct IsNotNullLegacyStatsRewrite;

impl StatsRewriteRule for IsNotNullLegacyStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        IsNotNull.id()
    }

    fn falsify(
        &self,
        expr: &Expression,
        _ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        Ok(null_count(expr.child(0))
            .map(|null_count| eq(null_count, RowCount.new_expr(EmptyOptions, []))))
    }

    fn satisfy(
        &self,
        expr: &Expression,
        _ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        Ok(null_count(expr.child(0)).map(|null_count| eq(null_count, lit(0u64))))
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
        expr: &Expression,
        _ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(all_null(expr.child(0))))
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
        expr: &Expression,
        _ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        Ok(Some(all_non_null(expr.child(0))))
    }
}

#[derive(Debug)]
struct ListContainsStatsRewrite;

impl StatsRewriteRule for ListContainsStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        ListContains.id()
    }

    fn falsify(
        &self,
        expr: &Expression,
        _ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        let list = expr.child(0);
        let needle = expr.child(1);

        let Some(list_scalar) = literal_stat(list, Stat::Min) else {
            return Ok(None);
        };
        let elements = list_scalar
            .as_opt::<Literal>()
            .and_then(|literal| literal.as_list_opt())
            .and_then(|list| list.elements());
        let Some(elements) = elements else {
            return Ok(None);
        };
        if elements.is_empty() {
            return Ok(Some(lit(true)));
        }

        let Some(value_max) = max(needle) else {
            return Ok(None);
        };
        let Some(value_min) = min(needle) else {
            return Ok(None);
        };

        Ok(and_collect(elements.iter().map(|value| {
            or(
                lt(value_max.clone(), lit(value.clone())),
                gt(value_min.clone(), lit(value.clone())),
            )
        })))
    }
}

fn min(expr: &Expression) -> Option<Expression> {
    stat_expr(expr, Stat::Min)
}

fn max(expr: &Expression) -> Option<Expression> {
    stat_expr(expr, Stat::Max)
}

fn bounded_min(expr: &Expression) -> Expression {
    if let Some(literal) = expr.as_opt::<Literal>() {
        return lit(literal.clone());
    }

    if let Some(dtype) = expr.as_opt::<Cast>() {
        return cast(bounded_min(expr.child(0)), dtype.clone());
    }

    stat_fn(
        expr.clone(),
        BoundedMin.bind(BoundedMinOptions {
            max_bytes: default_bounded_stat_max_bytes(),
        }),
    )
}

fn bounded_max(expr: &Expression) -> Expression {
    if let Some(literal) = expr.as_opt::<Literal>() {
        return lit(literal.clone());
    }

    if let Some(dtype) = expr.as_opt::<Cast>() {
        return cast(bounded_max(expr.child(0)), dtype.clone());
    }

    stat_fn(
        expr.clone(),
        BoundedMax.bind(BoundedMaxOptions {
            max_bytes: default_bounded_stat_max_bytes(),
        }),
    )
}

fn null_count(expr: &Expression) -> Option<Expression> {
    stat_expr(expr, Stat::NullCount)
}

fn all_null(expr: &Expression) -> Expression {
    stat_fn(expr.clone(), AllNull.bind(AggregateEmptyOptions))
}

fn all_non_null(expr: &Expression) -> Expression {
    stat_fn(expr.clone(), AllNonNull.bind(AggregateEmptyOptions))
}

fn stat_expr(expr: &Expression, stat: Stat) -> Option<Expression> {
    if let Some(literal) = literal_stat(expr, stat) {
        return Some(literal);
    }

    if let Some(dtype) = expr.as_opt::<Cast>() {
        return cast_stat(expr.child(0), dtype, stat);
    }

    stat.aggregate_fn()
        .map(|aggregate_fn| stat_fn(expr.clone(), aggregate_fn))
}

fn literal_stat(expr: &Expression, stat: Stat) -> Option<Expression> {
    let scalar = expr.as_opt::<Literal>()?;
    match stat {
        Stat::Min | Stat::Max => Some(lit(scalar.clone())),
        Stat::NullCount => Some(lit(if scalar.is_null() { 1u64 } else { 0u64 })),
        Stat::IsConstant
        | Stat::IsSorted
        | Stat::IsStrictSorted
        | Stat::Sum
        | Stat::UncompressedSizeInBytes
        | Stat::NaNCount => None,
    }
}

fn cast_stat(expr: &Expression, dtype: &DType, stat: Stat) -> Option<Expression> {
    match stat {
        Stat::Min | Stat::Max => stat_expr(expr, stat).map(|stat| cast(stat, dtype.clone())),
        Stat::NaNCount | Stat::Sum | Stat::UncompressedSizeInBytes => stat_expr(expr, stat),
        Stat::NullCount | Stat::IsConstant | Stat::IsSorted | Stat::IsStrictSorted => None,
    }
}

fn stat_fn(expr: Expression, aggregate_fn: AggregateFnRef) -> Expression {
    StatFn.new_expr(StatOptions::new(aggregate_fn), [expr])
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use super::StatFn;
    use super::StatOptions;
    use super::all_non_null;
    use super::all_null;
    use super::bounded_max;
    use super::bounded_min;
    use crate::aggregate_fn::AggregateFnRef;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::Expression;
    use crate::expr::and;
    use crate::expr::between;
    use crate::expr::cast;
    use crate::expr::col;
    use crate::expr::eq;
    use crate::expr::gt;
    use crate::expr::gt_eq;
    use crate::expr::is_not_null;
    use crate::expr::is_null;
    use crate::expr::list_contains;
    use crate::expr::lit;
    use crate::expr::lt;
    use crate::expr::lt_eq;
    use crate::expr::or;
    use crate::expr::stats::Stat;
    use crate::scalar::Scalar;
    use crate::scalar_fn::EmptyOptions;
    use crate::scalar_fn::ScalarFnVTableExt;
    use crate::scalar_fn::fns::between::BetweenOptions;
    use crate::scalar_fn::fns::between::StrictComparison;
    use crate::scalar_fn::internal::row_count::RowCount;
    use crate::stats::session::StatsSession;

    static SESSION: LazyLock<VortexSession> =
        LazyLock::new(|| VortexSession::empty().with::<StatsSession>());

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
            ]),
            Nullability::NonNullable,
        )
    }

    fn falsify(expr: &Expression) -> VortexResult<Option<Expression>> {
        expr.falsify(&test_scope(), &SESSION)
    }

    fn satisfy(expr: &Expression) -> VortexResult<Option<Expression>> {
        expr.satisfy(&test_scope(), &SESSION)
    }

    #[test]
    fn rewrites_comparison_falsifier() -> VortexResult<()> {
        let expr = gt(col("a"), lit(10));
        assert_eq!(
            falsify(&expr)?,
            Some(or(
                lt_eq(stat(col("a"), Stat::Max), lit(10)),
                lt_eq(bounded_max(&col("a")), lit(10)),
            ))
        );

        let expr = eq(col("a"), col("b"));
        assert_eq!(
            falsify(&expr)?,
            Some(or(
                or(
                    gt(stat(col("a"), Stat::Min), stat(col("b"), Stat::Max)),
                    gt(stat(col("b"), Stat::Min), stat(col("a"), Stat::Max)),
                ),
                or(
                    gt(bounded_min(&col("a")), bounded_max(&col("b"))),
                    gt(bounded_min(&col("b")), bounded_max(&col("a"))),
                ),
            ))
        );
        Ok(())
    }

    #[test]
    fn rewrites_boolean_falsifiers() -> VortexResult<()> {
        let expr = and(gt(col("a"), lit(10)), lt(col("a"), lit(50)));
        assert_eq!(
            falsify(&expr)?,
            Some(or(
                or(
                    lt_eq(stat(col("a"), Stat::Max), lit(10)),
                    lt_eq(bounded_max(&col("a")), lit(10)),
                ),
                or(
                    gt_eq(stat(col("a"), Stat::Min), lit(50)),
                    gt_eq(bounded_min(&col("a")), lit(50)),
                ),
            ))
        );
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

        assert_eq!(
            falsify(&expr)?,
            Some(or(
                or(
                    gt(lit(10), stat(col("a"), Stat::Max)),
                    gt(lit(10), bounded_max(&col("a"))),
                ),
                or(
                    gt(stat(col("a"), Stat::Min), lit(50)),
                    gt(bounded_min(&col("a")), lit(50)),
                ),
            ))
        );
        Ok(())
    }

    #[test]
    fn rewrites_null_falsifiers() -> VortexResult<()> {
        assert_eq!(
            falsify(&is_null(col("a")))?,
            Some(or(
                eq(stat(col("a"), Stat::NullCount), lit(0u64)),
                all_non_null(&col("a")),
            ))
        );

        assert_eq!(
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
        assert_eq!(
            satisfy(&is_null(col("a")))?,
            Some(or(
                eq(
                    stat(col("a"), Stat::NullCount),
                    RowCount.new_expr(EmptyOptions, []),
                ),
                all_null(&col("a")),
            ))
        );

        assert_eq!(
            satisfy(&is_not_null(col("a")))?,
            Some(or(
                eq(stat(col("a"), Stat::NullCount), lit(0u64)),
                all_non_null(&col("a")),
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

        assert_eq!(
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
    fn forwards_min_max_through_safe_cast() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I64, Nullability::NonNullable);
        let expr = eq(cast(col("a"), dtype.clone()), lit(42i64));

        assert_eq!(
            falsify(&expr)?,
            Some(or(
                or(
                    gt(cast(stat(col("a"), Stat::Min), dtype.clone()), lit(42i64)),
                    gt(lit(42i64), cast(stat(col("a"), Stat::Max), dtype.clone())),
                ),
                or(
                    gt(cast(bounded_min(&col("a")), dtype.clone()), lit(42i64)),
                    gt(lit(42i64), cast(bounded_max(&col("a")), dtype)),
                ),
            ))
        );
        Ok(())
    }
}
