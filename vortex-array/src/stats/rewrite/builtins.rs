// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use vortex_error::VortexResult;

use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::AggregateFnVTableExt;
use crate::aggregate_fn::EmptyOptions as AggregateEmptyOptions;
use crate::aggregate_fn::fns::all_non_nan::AllNonNan;
use crate::aggregate_fn::fns::all_non_null::AllNonNull;
use crate::aggregate_fn::fns::all_null::AllNull;
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
use crate::scalar::StringLike;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ScalarFnId;
use crate::scalar_fn::ScalarFnVTable;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::between::Between;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::cast::Cast;
use crate::scalar_fn::fns::dynamic::DynamicComparison;
use crate::scalar_fn::fns::dynamic::DynamicComparisonExpr;
use crate::scalar_fn::fns::is_not_null::IsNotNull;
use crate::scalar_fn::fns::is_null::IsNull;
use crate::scalar_fn::fns::like::Like;
use crate::scalar_fn::fns::like::LikeVariant;
use crate::scalar_fn::fns::list_contains::ListContains;
use crate::scalar_fn::fns::literal::Literal;
use crate::scalar_fn::fns::operators::CompareOperator;
use crate::scalar_fn::fns::operators::Operator;
use crate::scalar_fn::internal::row_count::RowCount;
use crate::stats::expr::StatFn;
use crate::stats::expr::StatOptions;
use crate::stats::rewrite::StatsRewriteCtx;
use crate::stats::rewrite::StatsRewriteRule;
use crate::stats::session::StatsSession;

/// Register built-in stats rewrite rules.
pub(crate) fn register_builtins(session: &StatsSession) {
    session.register_rewrite(BinaryStatsRewrite);
    session.register_rewrite(BetweenStatsRewrite);
    session.register_rewrite(IsNullLegacyStatsRewrite);
    session.register_rewrite(IsNullAllNonNullStatsRewrite);
    session.register_rewrite(IsNullAllNullStatsRewrite);
    session.register_rewrite(IsNotNullLegacyStatsRewrite);
    session.register_rewrite(IsNotNullAllNullStatsRewrite);
    session.register_rewrite(IsNotNullAllNonNullStatsRewrite);
    session.register_rewrite(LikeStatsRewrite);
    session.register_rewrite(ListContainsStatsRewrite);
    session.register_rewrite(DynamicComparisonStatsRewrite);
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
                    .map(|value_predicate| with_nan_predicate(ctx, lhs, rhs, value_predicate))
                    .transpose()?
            }
            Operator::NotEq => min(lhs)
                .zip(max(rhs))
                .zip(max(lhs).zip(min(rhs)))
                .map(|((min_lhs, max_rhs), (max_lhs, min_rhs))| {
                    with_nan_predicate(
                        ctx,
                        lhs,
                        rhs,
                        and(eq(min_lhs, max_rhs), eq(max_lhs, min_rhs)),
                    )
                })
                .transpose()?,
            Operator::Gt => max(lhs)
                .zip(min(rhs))
                .map(|(a, b)| with_nan_predicate(ctx, lhs, rhs, lt_eq(a, b)))
                .transpose()?,
            Operator::Gte => max(lhs)
                .zip(min(rhs))
                .map(|(a, b)| with_nan_predicate(ctx, lhs, rhs, lt(a, b)))
                .transpose()?,
            Operator::Lt => min(lhs)
                .zip(max(rhs))
                .map(|(a, b)| with_nan_predicate(ctx, lhs, rhs, gt_eq(a, b)))
                .transpose()?,
            Operator::Lte => min(lhs)
                .zip(max(rhs))
                .map(|(a, b)| with_nan_predicate(ctx, lhs, rhs, gt(a, b)))
                .transpose()?,
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
struct LikeStatsRewrite;

impl StatsRewriteRule for LikeStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Like.id()
    }

    fn falsify(
        &self,
        expr: &Expression,
        _ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
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
                min(source)
                    .zip(max(source))
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
                min(source)
                    .zip(max(source))
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
struct ListContainsStatsRewrite;

impl StatsRewriteRule for ListContainsStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        ListContains.id()
    }

    fn falsify(
        &self,
        expr: &Expression,
        ctx: &StatsRewriteCtx<'_>,
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

        let value_predicate = and_collect(elements.iter().map(|value| {
            or(
                lt(value_max.clone(), lit(value.clone())),
                gt(value_min.clone(), lit(value.clone())),
            )
        }));
        value_predicate
            .map(|value_predicate| with_all_non_nan_predicate(ctx, [needle], value_predicate))
            .transpose()
    }
}

#[derive(Debug)]
struct DynamicComparisonStatsRewrite;

impl StatsRewriteRule for DynamicComparisonStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        DynamicComparison.id()
    }

    fn falsify(
        &self,
        expr: &Expression,
        ctx: &StatsRewriteCtx<'_>,
    ) -> VortexResult<Option<Expression>> {
        let dynamic = expr.as_::<DynamicComparison>();
        let lhs = expr.child(0);

        let Some((operator, lhs_stat)) = (match dynamic.operator {
            CompareOperator::Eq | CompareOperator::NotEq => None,
            CompareOperator::Gt => max(lhs).map(|lhs_stat| (CompareOperator::Lte, lhs_stat)),
            CompareOperator::Gte => max(lhs).map(|lhs_stat| (CompareOperator::Lt, lhs_stat)),
            CompareOperator::Lt => min(lhs).map(|lhs_stat| (CompareOperator::Gte, lhs_stat)),
            CompareOperator::Lte => min(lhs).map(|lhs_stat| (CompareOperator::Gt, lhs_stat)),
        }) else {
            return Ok(None);
        };

        let value_predicate = DynamicComparison.new_expr(
            DynamicComparisonExpr {
                operator,
                rhs: Arc::clone(&dynamic.rhs),
                default: !dynamic.default,
            },
            [lhs_stat],
        );
        with_all_non_nan_predicate(ctx, [lhs], value_predicate).map(Some)
    }
}

fn min(expr: &Expression) -> Option<Expression> {
    stat_expr(expr, Stat::Min)
}

fn max(expr: &Expression) -> Option<Expression> {
    stat_expr(expr, Stat::Max)
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

// Min/max do not order NaN values, so comparison rewrites are only sound when every
// candidate value is known to be non-NaN. Cast result dtypes are not enough: a cast
// from float to non-float still needs a proof about the float source values.
fn all_non_nan_stat(
    ctx: &StatsRewriteCtx<'_>,
    expr: &Expression,
) -> VortexResult<Option<Expression>> {
    if let Some(scalar) = expr.as_opt::<Literal>() {
        let Some(value) = scalar.as_primitive_opt() else {
            return Ok(None);
        };
        return Ok(value.is_nan().then(|| lit(false)));
    }

    if expr.is::<Cast>() {
        if !has_nans(&ctx.return_dtype(expr.child(0))?) {
            return Ok(None);
        }

        return all_non_nan_stat(ctx, expr.child(0));
    }

    if !has_nans(&ctx.return_dtype(expr)?) {
        return Ok(None);
    }

    Ok(Some(stat_fn(
        expr.clone(),
        AllNonNan.bind(AggregateEmptyOptions),
    )))
}

fn has_nans(dtype: &DType) -> bool {
    matches!(dtype, DType::Primitive(ptype, _) if ptype.is_float())
}

fn stat_expr(expr: &Expression, stat: Stat) -> Option<Expression> {
    if let Some(literal) = literal_stat(expr, stat) {
        return Some(literal);
    }

    // `literal_stat` handled every stat that is defined for literals. If it returned
    // `None`, the requested stat is not meaningful for this literal, such as
    // `NaNCount` over a non-float value, so do not manufacture `stat(literal, ...)`.
    if expr.is::<Literal>() {
        return None;
    }

    if let Some(dtype) = expr.as_opt::<Cast>() {
        return cast_stat(expr.child(0), dtype, stat);
    }

    stat.aggregate_fn()
        .map(|aggregate_fn| stat_fn(expr.clone(), aggregate_fn))
}

fn with_nan_predicate(
    ctx: &StatsRewriteCtx<'_>,
    lhs: &Expression,
    rhs: &Expression,
    value_predicate: Expression,
) -> VortexResult<Expression> {
    with_all_non_nan_predicate(ctx, [lhs, rhs], value_predicate)
}

fn with_all_non_nan_predicate<'a>(
    ctx: &StatsRewriteCtx<'_>,
    exprs: impl IntoIterator<Item = &'a Expression>,
    value_predicate: Expression,
) -> VortexResult<Expression> {
    let mut nan_checks = Vec::new();
    for expr in exprs {
        if let Some(check) = all_non_nan_stat(ctx, expr)? {
            nan_checks.push(check);
        }
    }
    let nan_predicate = and_collect(nan_checks);

    Ok(match nan_predicate {
        Some(nan_check) => and(nan_check, value_predicate),
        // No possible NaN-bearing expression remains, so the value predicate is
        // already guarded.
        None => value_predicate,
    })
}

fn literal_stat(expr: &Expression, stat: Stat) -> Option<Expression> {
    let scalar = expr.as_opt::<Literal>()?;
    match stat {
        Stat::Min | Stat::Max => Some(lit(scalar.clone())),
        Stat::NullCount => Some(lit(if scalar.is_null() { 1u64 } else { 0u64 })),
        Stat::NaNCount => {
            let value = scalar.as_primitive_opt()?;
            if !value.ptype().is_float() {
                return None;
            }

            Some(lit(if value.is_nan() { 1u64 } else { 0u64 }))
        }
        Stat::IsConstant
        | Stat::IsSorted
        | Stat::IsStrictSorted
        | Stat::Sum
        | Stat::UncompressedSizeInBytes => None,
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
    use crate::aggregate_fn::AggregateFnRef;
    use crate::aggregate_fn::AggregateFnVTableExt;
    use crate::aggregate_fn::EmptyOptions as AggregateEmptyOptions;
    use crate::aggregate_fn::fns::all_non_nan::AllNonNan;
    use crate::dtype::DType;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::Expression;
    use crate::expr::and;
    use crate::expr::between;
    use crate::expr::cast;
    use crate::expr::col;
    use crate::expr::dynamic;
    use crate::expr::eq;
    use crate::expr::gt;
    use crate::expr::gt_eq;
    use crate::expr::is_not_null;
    use crate::expr::is_null;
    use crate::expr::like;
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
    use crate::scalar_fn::fns::dynamic::DynamicComparison;
    use crate::scalar_fn::fns::dynamic::DynamicComparisonExpr;
    use crate::scalar_fn::fns::operators::CompareOperator;
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
                ("f", DType::Primitive(PType::F32, Nullability::NonNullable)),
                ("s", DType::Utf8(Nullability::NonNullable)),
                ("t", DType::Utf8(Nullability::NonNullable)),
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

    fn nan_free(expr: Expression) -> Expression {
        stat_fn(expr, AllNonNan.bind(AggregateEmptyOptions))
    }

    #[test]
    fn rewrites_comparison_falsifier() -> VortexResult<()> {
        let expr = gt(col("a"), lit(10));
        assert_eq!(
            falsify(&expr)?,
            Some(lt_eq(stat(col("a"), Stat::Max), lit(10)))
        );

        let expr = eq(col("a"), col("b"));
        assert_eq!(
            falsify(&expr)?,
            Some(or(
                gt(stat(col("a"), Stat::Min), stat(col("b"), Stat::Max)),
                gt(stat(col("b"), Stat::Min), stat(col("a"), Stat::Max)),
            ))
        );

        let expr = eq(col("s"), col("t"));
        assert_eq!(
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
        assert_eq!(
            falsify(&expr)?,
            Some(or(
                lt_eq(stat(col("a"), Stat::Max), lit(10)),
                gt_eq(stat(col("a"), Stat::Min), lit(50)),
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
                gt(lit(10), stat(col("a"), Stat::Max)),
                gt(stat(col("a"), Stat::Min), lit(50)),
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
    fn rewrites_like_falsifier() -> VortexResult<()> {
        let expr = like(col("s"), lit("prefix%"));
        assert_eq!(
            falsify(&expr)?,
            Some(or(
                gt_eq(stat(col("s"), Stat::Min), lit("prefiy")),
                lt(stat(col("s"), Stat::Max), lit("prefix")),
            ))
        );

        let expr = like(col("s"), lit("exact"));
        assert_eq!(
            falsify(&expr)?,
            Some(or(
                gt(stat(col("s"), Stat::Min), lit("exact")),
                lt(stat(col("s"), Stat::Max), lit("exact")),
            ))
        );

        let expr = like(col("s"), lit("%suffix"));
        assert_eq!(falsify(&expr)?, None);
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

        assert_eq!(
            falsify(&expr)?,
            Some(DynamicComparison.new_expr(
                DynamicComparisonExpr {
                    operator: CompareOperator::Lte,
                    rhs: Arc::clone(&dynamic.rhs),
                    default: false,
                },
                [stat(col("a"), Stat::Max)],
            ))
        );
        Ok(())
    }

    #[test]
    fn nan_guard_tracks_cast_source_dtype() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let expr = gt(cast(col("f"), dtype.clone()), lit(5i32));

        assert_eq!(
            falsify(&expr)?,
            Some(and(
                nan_free(col("f")),
                lt_eq(cast(stat(col("f"), Stat::Max), dtype), lit(5i32)),
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
                gt(cast(stat(col("a"), Stat::Min), dtype.clone()), lit(42i64)),
                gt(lit(42i64), cast(stat(col("a"), Stat::Max), dtype)),
            ))
        );
        Ok(())
    }
}
