// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Statistics-based pruning rewrites for this crate's scalar functions.

use std::cmp::Ordering;

use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::and;
use vortex_array::expr::bound::and_collect;
use vortex_array::expr::bound::binary;
use vortex_array::expr::bound::eq;
use vortex_array::expr::bound::gt;
use vortex_array::expr::bound::gt_eq;
use vortex_array::expr::bound::lit;
use vortex_array::expr::bound::lt;
use vortex_array::expr::bound::or;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar::Scalar;
use vortex_array::scalar::StringLike;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::builtin::literal::Literal;
use vortex_array::scalar_fn::builtin::operators::CompareOperator;
use vortex_error::VortexResult;
use vortex_expr::stats::rewrite::StatsRewriteRule;
use vortex_expr::stats::rewrite::falsify;
use vortex_expr::stats::rewrite::helpers::*;
use vortex_expr::stats::session::StatsSessionExt;
use vortex_session::VortexSession;

use self::like_variant::LikeVariant;
use crate::between::Between;
use crate::dynamic::DynamicComparison;
use crate::expr::bound::dynamic_with_options;
use crate::is_nan::IsNan;
use crate::like::Like;
use crate::list_contains::ListContains;

mod like_variant;

/// Register the stats rewrite rules for this crate's scalar functions.
pub fn initialize(session: &VortexSession) {
    let stats = session.stats();
    stats.register_rewrite(BetweenStatsRewrite);
    stats.register_rewrite(IsNanNaNCountStatsRewrite);
    stats.register_rewrite(IsNanAllNonNanStatsRewrite);
    stats.register_rewrite(IsNanAllNanStatsRewrite);
    stats.register_rewrite(LikeStatsRewrite);
    stats.register_rewrite(ListContainsNanCountStatsRewrite);
    stats.register_rewrite(ListContainsAllNonNanStatsRewrite);
    stats.register_rewrite(DynamicComparisonNanCountStatsRewrite);
    stats.register_rewrite(DynamicComparisonAllNonNanStatsRewrite);
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

        let lhs = binary(options.lower_strict.to_operator(), lower, arr.clone());
        let rhs = binary(options.upper_strict.to_operator(), arr, upper);
        falsify(&and(lhs, rhs), session)
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
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(nan_count(expr.child(0)).map(|nan_count| eq(nan_count, lit(0u64))))
    }

    fn satisfy(
        &self,
        expr: &BoundExpression,
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(nan_count(expr.child(0)).map(|nan_count| eq(nan_count, row_count())))
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
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(Some(all_non_nan(expr.child(0))))
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
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(Some(all_nan(expr.child(0))))
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
        _session: &VortexSession,
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
struct ListContainsNanCountStatsRewrite;

impl StatsRewriteRule for ListContainsNanCountStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        ListContains.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        list_contains_falsify::<NanCountProof>(expr)
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
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        list_contains_falsify::<AllNonNanProof>(expr)
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
) -> VortexResult<Option<BoundExpression>> {
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
        return Ok(P::EMIT_UNGUARDED_REWRITES.then(|| lit(true)));
    }

    let Some(value_max) = max(needle) else {
        return Ok(None);
    };
    let Some(value_min) = min(needle) else {
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
        .map(|value_predicate| with_non_nan_guards::<P>([needle], value_predicate))
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
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        dynamic_comparison_falsify::<NanCountProof>(expr)
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
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        dynamic_comparison_falsify::<AllNonNanProof>(expr)
    }
}

fn dynamic_comparison_falsify<P: NonNanProof>(
    expr: &BoundExpression,
) -> VortexResult<Option<BoundExpression>> {
    let dynamic = expr.as_::<DynamicComparison>();
    let lhs = expr.child(0);

    let Some((operator, lhs_stat)) = (match dynamic.operator() {
        CompareOperator::Eq | CompareOperator::NotEq => None,
        CompareOperator::Gt => max(lhs).map(|lhs_stat| (CompareOperator::Lte, lhs_stat)),
        CompareOperator::Gte => max(lhs).map(|lhs_stat| (CompareOperator::Lt, lhs_stat)),
        CompareOperator::Lt => min(lhs).map(|lhs_stat| (CompareOperator::Gte, lhs_stat)),
        CompareOperator::Lte => min(lhs).map(|lhs_stat| (CompareOperator::Gt, lhs_stat)),
    }) else {
        return Ok(None);
    };

    let value_predicate = dynamic_with_options(
        dynamic.with_operator(operator, !dynamic.default()),
        lhs_stat,
    );
    with_non_nan_guards::<P>([lhs], value_predicate)
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;

    use vortex_array::aggregate_fn::AggregateFnRef;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::BoundExpression;
    use vortex_array::expr::Expression;
    use vortex_array::expr::and;
    use vortex_array::expr::col;
    use vortex_array::expr::eq;
    use vortex_array::expr::get_item;
    use vortex_array::expr::gt;
    use vortex_array::expr::gt_eq;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt;
    use vortex_array::expr::or;
    use vortex_array::expr::stats::Stat;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnVTableExt;
    use vortex_array::scalar_fn::builtin::operators::CompareOperator;
    use vortex_array::scalar_fn::internal::row_count::RowCount;
    use vortex_error::VortexResult;
    use vortex_expr::stats::expr::StatFn;
    use vortex_expr::stats::expr::StatOptions;
    use vortex_expr::stats::rewrite::BoundExpressionPruneExt;
    use vortex_session::VortexSession;

    use crate::between::BetweenOptions;
    use crate::between::StrictComparison;
    use crate::dynamic::DynamicComparison;
    use crate::expr::between;
    use crate::expr::dynamic;
    use crate::expr::is_nan;
    use crate::expr::like;
    use crate::expr::list_contains;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

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

    fn all_nan(expr: &Expression) -> Expression {
        vortex_expr::stats::all_nan(expr.clone())
    }

    fn all_non_nan(expr: &Expression) -> Expression {
        vortex_expr::stats::all_non_nan(expr.clone())
    }

    macro_rules! assert_rewrite_eq {
        ($actual:expr, $expected:expr) => {
            assert_eq!($actual, bind_expected($expected)?)
        };
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

        // NaN literals fold through the nan_count stat.
        assert_rewrite_eq!(
            falsify(&is_nan(lit(f32::NAN)))?,
            Some(or(eq(lit(1u64), lit(0u64)), all_non_nan(&lit(f32::NAN))))
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
                all_nan(&lit(f32::NAN)),
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
    fn list_falsification() -> VortexResult<()> {
        let expr = list_contains(
            lit(Scalar::list(
                Arc::new(DType::Primitive(PType::I32, Nullability::NonNullable)),
                vec![1.into(), 2.into(), 3.into()],
                Nullability::NonNullable,
            )),
            col("a"),
        );
        let scope = DType::Struct(
            StructFields::new(
                ["a"].into(),
                vec![DType::Primitive(PType::I32, Nullability::NonNullable)],
            ),
            Nullability::NonNullable,
        );

        assert_eq!(
            expr.bind(&scope)?.falsify(&SESSION)?,
            Some(
                and(
                    and(
                        or(
                            lt(stat(col("a"), Stat::Max), lit(1i32)),
                            gt(stat(col("a"), Stat::Min), lit(1i32)),
                        ),
                        or(
                            lt(stat(col("a"), Stat::Max), lit(2i32)),
                            gt(stat(col("a"), Stat::Min), lit(2i32)),
                        )
                    ),
                    or(
                        lt(stat(col("a"), Stat::Max), lit(3i32)),
                        gt(stat(col("a"), Stat::Min), lit(3i32)),
                    )
                )
                .bind(&scope)?
            )
        );
        Ok(())
    }
}
