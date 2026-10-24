// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::and;
use vortex_array::expr::bound::eq;
use vortex_array::expr::bound::gt;
use vortex_array::expr::bound::gt_eq;
use vortex_array::expr::bound::lit;
use vortex_array::expr::bound::lt;
use vortex_array::expr::bound::lt_eq;
use vortex_array::expr::bound::or_collect;
use vortex_array::scalar_fn::ScalarFnId;
use vortex_array::scalar_fn::ScalarFnVTable;
use vortex_array::scalar_fn::builtin::binary::Binary;
use vortex_array::scalar_fn::builtin::is_not_null::IsNotNull;
use vortex_array::scalar_fn::builtin::is_null::IsNull;
use vortex_array::scalar_fn::builtin::literal::Literal;
use vortex_array::scalar_fn::builtin::operators::Operator;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::stats::rewrite::StatsRewriteRule;
use crate::stats::rewrite::falsify;
use crate::stats::rewrite::helpers::*;
use crate::stats::session::StatsSession;

/// Register the stats rewrite rules for the scalar functions vortex-array is built on.
pub(crate) fn register_builtins(session: &StatsSession) {
    session.register_rewrite(BinaryNanCountStatsRewrite);
    session.register_rewrite(BinaryAllNonNanStatsRewrite);
    session.register_rewrite(LiteralStatsRewrite);
    session.register_rewrite(IsNullNullCountStatsRewrite);
    session.register_rewrite(IsNullAllNonNullStatsRewrite);
    session.register_rewrite(IsNullAllNullStatsRewrite);
    session.register_rewrite(IsNotNullNullCountStatsRewrite);
    session.register_rewrite(IsNotNullAllNullStatsRewrite);
    session.register_rewrite(IsNotNullAllNonNullStatsRewrite);
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

/// Bool literal predicate value
fn literal_truth(expr: &BoundExpression) -> Option<bool> {
    expr.as_::<Literal>().as_bool_opt().and_then(|b| b.value())
}

#[derive(Debug)]
struct BinaryNanCountStatsRewrite;

impl StatsRewriteRule for BinaryNanCountStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Binary.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        binary_falsify::<NanCountProof>(expr, session)
    }
}

#[derive(Debug)]
struct BinaryAllNonNanStatsRewrite;

impl StatsRewriteRule for BinaryAllNonNanStatsRewrite {
    fn scalar_fn_id(&self) -> ScalarFnId {
        Binary.id()
    }

    fn falsify(
        &self,
        expr: &BoundExpression,
        session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        binary_falsify::<AllNonNanProof>(expr, session)
    }
}

fn binary_falsify<P: NonNanProof>(
    expr: &BoundExpression,
    session: &VortexSession,
) -> VortexResult<Option<BoundExpression>> {
    let operator = expr.as_::<Binary>();
    let lhs = expr.child(0);
    let rhs = expr.child(1);

    Ok(match operator {
        Operator::Eq => {
            let left = min(lhs).zip(max(rhs)).map(|(a, b)| gt(a, b));
            let right = min(rhs).zip(max(lhs)).map(|(a, b)| gt(a, b));
            or_collect(left.into_iter().chain(right))
                .map(|value_predicate| with_non_nan_guards::<P>([lhs, rhs], value_predicate))
                .transpose()?
                .flatten()
        }
        Operator::NotEq => min(lhs)
            .zip(max(rhs))
            .zip(max(lhs).zip(min(rhs)))
            .map(|((min_lhs, max_rhs), (max_lhs, min_rhs))| {
                with_non_nan_guards::<P>(
                    [lhs, rhs],
                    and(eq(min_lhs, max_rhs), eq(max_lhs, min_rhs)),
                )
            })
            .transpose()?
            .flatten(),
        Operator::Gt => max(lhs)
            .zip(min(rhs))
            .map(|(a, b)| with_non_nan_guards::<P>([lhs, rhs], lt_eq(a, b)))
            .transpose()?
            .flatten(),
        Operator::Gte => max(lhs)
            .zip(min(rhs))
            .map(|(a, b)| with_non_nan_guards::<P>([lhs, rhs], lt(a, b)))
            .transpose()?
            .flatten(),
        Operator::Lt => min(lhs)
            .zip(max(rhs))
            .map(|(a, b)| with_non_nan_guards::<P>([lhs, rhs], gt_eq(a, b)))
            .transpose()?
            .flatten(),
        Operator::Lte => min(lhs)
            .zip(max(rhs))
            .map(|(a, b)| with_non_nan_guards::<P>([lhs, rhs], gt(a, b)))
            .transpose()?
            .flatten(),
        Operator::And => {
            if !P::EMIT_UNGUARDED_REWRITES {
                return Ok(None);
            }

            let lhs_falsifier = falsify(lhs, session)?;
            let rhs_falsifier = falsify(rhs, session)?;
            or_collect(lhs_falsifier.into_iter().chain(rhs_falsifier))
        }
        Operator::Or => {
            // Check before recursing: falsifying the children first would repeat the
            // whole recursive rewrite once per registered `Binary` rule, which is
            // exponential in `Or`-nesting depth for chains like `a = 1 OR a = 2 OR ...`.
            if !P::EMIT_UNGUARDED_REWRITES {
                return Ok(None);
            }

            match (falsify(lhs, session)?, falsify(rhs, session)?) {
                (Some(lhs), Some(rhs)) => Some(and(lhs, rhs)),
                _ => None,
            }
        }
        Operator::Add | Operator::Sub | Operator::Mul | Operator::Div => None,
    })
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
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(null_count(expr.child(0)).map(|null_count| eq(null_count, lit(0u64))))
    }

    fn satisfy(
        &self,
        expr: &BoundExpression,
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(null_count(expr.child(0)).map(|null_count| eq(null_count, row_count())))
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
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
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
        expr: &BoundExpression,
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(Some(all_null(expr.child(0))))
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
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(null_count(expr.child(0)).map(|null_count| eq(null_count, row_count())))
    }

    fn satisfy(
        &self,
        expr: &BoundExpression,
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
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
        expr: &BoundExpression,
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
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
        expr: &BoundExpression,
        _session: &VortexSession,
    ) -> VortexResult<Option<BoundExpression>> {
        Ok(Some(all_non_null(expr.child(0))))
    }
}

#[cfg(test)]
#[cfg(test)]
mod tests {
    use std::sync::Arc;
    use std::sync::LazyLock;
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use vortex_array::aggregate_fn::AggregateFnRef;
    use vortex_array::aggregate_fn::AggregateFnVTableExt;
    use vortex_array::aggregate_fn::EmptyOptions as AggregateEmptyOptions;
    use vortex_array::aggregate_fn::fns::all_non_nan::AllNonNan;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::StructFields;
    use vortex_array::expr::BoundExpression;
    use vortex_array::expr::Expression;
    use vortex_array::expr::and;
    use vortex_array::expr::cast;
    use vortex_array::expr::col;
    use vortex_array::expr::eq;
    use vortex_array::expr::gt;
    use vortex_array::expr::gt_eq;
    use vortex_array::expr::is_not_null;
    use vortex_array::expr::is_null;
    use vortex_array::expr::lit;
    use vortex_array::expr::lt;
    use vortex_array::expr::lt_eq;
    use vortex_array::expr::or;
    use vortex_array::expr::stats::Stat;
    use vortex_array::scalar::Scalar;
    use vortex_array::scalar_fn::EmptyOptions;
    use vortex_array::scalar_fn::ScalarFnId;
    use vortex_array::scalar_fn::ScalarFnVTable;
    use vortex_array::scalar_fn::ScalarFnVTableExt;
    use vortex_array::scalar_fn::builtin::binary::Binary;
    use vortex_array::scalar_fn::internal::row_count::RowCount;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

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

    /// Counts how many times the stats rewrite visits a `Binary` node.
    #[derive(Debug)]
    struct BinaryVisitCounter(Arc<AtomicUsize>);

    impl StatsRewriteRule for BinaryVisitCounter {
        fn scalar_fn_id(&self) -> ScalarFnId {
            Binary.id()
        }

        fn falsify(
            &self,
            _expr: &BoundExpression,
            _session: &VortexSession,
        ) -> VortexResult<Option<BoundExpression>> {
            self.0.fetch_add(1, Ordering::Relaxed);
            Ok(None)
        }
    }

    #[test]
    fn or_chain_falsify_visits_each_node_once() -> VortexResult<()> {
        let session = VortexSession::empty();
        let visits = Arc::new(AtomicUsize::new(0));
        session
            .stats()
            .register_rewrite(BinaryVisitCounter(Arc::clone(&visits)));

        let expr = (1..16).fold(eq(col("a"), lit(0)), |chain, i| {
            or(chain, eq(col("a"), lit(i)))
        });
        let falsifier = expr.bind(&test_scope())?.falsify(&session)?;
        assert!(falsifier.is_some());

        // One visit per `Binary` node: 16 comparisons plus 15 `or`s.
        assert_eq!(visits.load(Ordering::Relaxed), 31);
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
