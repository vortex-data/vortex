// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bind abstract `vortex.stat` expressions to a concrete stats representation.
//!
//! Stats rewrite rules describe pruning in terms of `vortex.stat(input, aggregate_fn)` placeholders
//! so the rewrite is independent of where statistics are stored. Binding is the later pass that
//! replaces those placeholders with the representation used by a caller: zone-map field references,
//! file-level stat literals, or typed nulls for missing stats.

use vortex_error::VortexResult;

use crate::aggregate_fn::AggregateFnRef;
use crate::aggregate_fn::fns::all_nan::AllNan;
use crate::aggregate_fn::fns::all_non_nan::AllNonNan;
use crate::aggregate_fn::fns::all_non_null::AllNonNull;
use crate::aggregate_fn::fns::all_null::AllNull;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::eq;
use crate::expr::lit;
use crate::expr::stats::Stat;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::Transformed;
use crate::scalar::Scalar;
use crate::scalar_fn::EmptyOptions;
use crate::scalar_fn::ScalarFnVTableExt;
use crate::scalar_fn::fns::stat::StatFn;
use crate::scalar_fn::internal::row_count::RowCount;

/// A target that can bind abstract statistics to concrete expressions.
pub trait StatBinder {
    /// The dtype scope used to type-check expressions before stats are bound.
    fn scope(&self) -> &DType;

    /// The dtype scope used after stats have been bound.
    ///
    /// Binders that rewrite stats to a different root expression, such as a
    /// stats-table row, should return that post-binding root dtype.
    fn bound_scope(&self) -> DType {
        self.scope().clone()
    }

    /// Bind `stat(input)` to a concrete expression.
    ///
    /// Returning `Ok(None)` marks the stat as unavailable. [`bind_stats`] will
    /// then call [`Self::missing_stat`] with the dtype expected from the
    /// original `vortex.stat` expression.
    fn bind_stat(
        &mut self,
        input: &Expression,
        stat: Stat,
        stat_dtype: &DType,
    ) -> VortexResult<Option<Expression>>;

    /// Bind `aggregate_fn(input)` to a concrete expression.
    ///
    /// The default implementation supports aggregate functions with legacy
    /// [`Stat`] slots. Binders that store richer aggregate stats can override
    /// this method without extending the generic stats binding walker.
    fn bind_aggregate(
        &mut self,
        input: &Expression,
        aggregate_fn: &AggregateFnRef,
        stat_dtype: &DType,
    ) -> VortexResult<Option<Expression>> {
        bind_direct_aggregate_stat(self, input, aggregate_fn, stat_dtype)
    }

    /// Bind one of the legacy stat slots for `input`.
    fn bind_legacy_stat(
        &mut self,
        input: &Expression,
        stat: Stat,
    ) -> VortexResult<Option<Expression>> {
        let input_dtype = input.return_dtype(self.scope())?;
        let Some(stat_dtype) = stat.dtype(&input_dtype) else {
            return Ok(None);
        };
        self.bind_stat(input, stat, &stat_dtype)
    }

    /// Expression to use when a stat is unavailable.
    ///
    /// The default is a nullable null literal, which preserves three-valued
    /// pruning semantics for stats-table execution.
    fn missing_stat(&mut self, dtype: DType) -> VortexResult<Expression> {
        Ok(null_expr(dtype))
    }
}

/// Bind all `vortex.stat` expressions in `predicate`.
///
/// The predicate is usually the output of a stats rewrite rule. Rewrite rules
/// are responsible for expressing stat semantics; binding maps aggregate-backed
/// stat requests to the concrete stats representation supported by the binder.
pub fn bind_stats(predicate: Expression, binder: &mut impl StatBinder) -> VortexResult<Expression> {
    let scope = binder.scope().clone();
    let lowered = predicate
        .transform_down(|expr| {
            if !expr.is::<StatFn>() {
                return Ok(Transformed::no(expr));
            }

            match bind_stat_fn(&expr, &scope, binder)? {
                Some(bound) => Ok(Transformed::yes(bound)),
                None => {
                    let dtype = expr.return_dtype(&scope)?;
                    Ok(Transformed::yes(binder.missing_stat(dtype)?))
                }
            }
        })?
        .into_inner();

    lowered.optimize_recursive(&binder.bound_scope())
}

/// Bind aggregate stats that can be derived from legacy count stat slots.
///
/// This is an opt-in helper for stats backends that materialize `NaNCount` and
/// `NullCount`, but do not materialize aggregate boolean stats directly.
pub fn bind_legacy_count_aggregate<B: StatBinder + ?Sized>(
    binder: &mut B,
    input: &Expression,
    aggregate_fn: &AggregateFnRef,
) -> VortexResult<Option<Expression>> {
    if aggregate_fn.is::<AllNan>() {
        let Some(nan_count) = binder.bind_legacy_stat(input, Stat::NaNCount)? else {
            return Ok(None);
        };
        return Ok(Some(eq(nan_count, RowCount.new_expr(EmptyOptions, []))));
    }

    if aggregate_fn.is::<AllNonNan>() {
        let Some(nan_count) = binder.bind_legacy_stat(input, Stat::NaNCount)? else {
            return Ok(None);
        };
        return Ok(Some(eq(nan_count, lit(0u64))));
    }

    if aggregate_fn.is::<AllNull>() {
        let Some(null_count) = binder.bind_legacy_stat(input, Stat::NullCount)? else {
            return Ok(None);
        };
        return Ok(Some(eq(null_count, RowCount.new_expr(EmptyOptions, []))));
    }

    if aggregate_fn.is::<AllNonNull>() {
        let Some(null_count) = binder.bind_legacy_stat(input, Stat::NullCount)? else {
            return Ok(None);
        };
        return Ok(Some(eq(null_count, lit(0u64))));
    }

    Ok(None)
}

/// Bind an aggregate function that has a direct legacy [`Stat`] slot.
pub fn bind_direct_aggregate_stat<B: StatBinder + ?Sized>(
    binder: &mut B,
    input: &Expression,
    aggregate_fn: &AggregateFnRef,
    stat_dtype: &DType,
) -> VortexResult<Option<Expression>> {
    let Some(stat) = Stat::from_aggregate_fn(aggregate_fn) else {
        return Ok(None);
    };
    binder.bind_stat(input, stat, stat_dtype)
}

/// Bind aggregate stats for backends that expose legacy count-derived stats.
///
/// Backends using this helper first bind aggregate facts derivable from
/// `NaNCount` and `NullCount`, then fall back to direct aggregate-to-stat
/// mappings.
pub fn bind_legacy_count_or_direct_aggregate<B: StatBinder + ?Sized>(
    binder: &mut B,
    input: &Expression,
    aggregate_fn: &AggregateFnRef,
    stat_dtype: &DType,
) -> VortexResult<Option<Expression>> {
    if let Some(bound) = bind_legacy_count_aggregate(binder, input, aggregate_fn)? {
        return Ok(Some(bound));
    }

    bind_direct_aggregate_stat(binder, input, aggregate_fn, stat_dtype)
}

fn bind_stat_fn(
    expr: &Expression,
    scope: &DType,
    binder: &mut impl StatBinder,
) -> VortexResult<Option<Expression>> {
    let options = expr.as_::<StatFn>();
    let aggregate_fn = options.aggregate_fn();
    // `StatFn` has exactly one child: the expression the aggregate statistic is computed over.
    let input = expr.child(0);

    let stat_dtype = expr.return_dtype(scope)?;
    binder.bind_aggregate(input, aggregate_fn, &stat_dtype)
}

fn null_expr(dtype: DType) -> Expression {
    lit(Scalar::null(dtype.as_nullable()))
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::dtype::Nullability;
    use crate::dtype::PType;
    use crate::dtype::StructFields;
    use crate::expr::and;
    use crate::expr::col;
    use crate::expr::get_item;
    use crate::expr::is_null;
    use crate::expr::or;
    use crate::expr::root;
    use crate::stats::all_non_nan;

    struct TestBinder {
        input_scope: DType,
        bound_scope: DType,
        bind_nan_count: bool,
    }

    impl TestBinder {
        fn new(bind_nan_count: bool) -> Self {
            Self {
                input_scope: DType::Struct(
                    StructFields::from_iter([(
                        "f",
                        DType::Primitive(PType::F32, Nullability::NonNullable),
                    )]),
                    Nullability::NonNullable,
                ),
                bound_scope: DType::Struct(
                    StructFields::from_iter([(
                        "f_nan_count",
                        DType::Primitive(PType::U64, Nullability::NonNullable),
                    )]),
                    Nullability::NonNullable,
                ),
                bind_nan_count,
            }
        }
    }

    impl StatBinder for TestBinder {
        fn scope(&self) -> &DType {
            &self.input_scope
        }

        fn bound_scope(&self) -> DType {
            self.bound_scope.clone()
        }

        fn bind_stat(
            &mut self,
            _input: &Expression,
            stat: Stat,
            _stat_dtype: &DType,
        ) -> VortexResult<Option<Expression>> {
            if stat == Stat::NaNCount && self.bind_nan_count {
                Ok(Some(get_item("f_nan_count", root())))
            } else {
                Ok(None)
            }
        }

        fn bind_aggregate(
            &mut self,
            input: &Expression,
            aggregate_fn: &AggregateFnRef,
            stat_dtype: &DType,
        ) -> VortexResult<Option<Expression>> {
            bind_legacy_count_or_direct_aggregate(self, input, aggregate_fn, stat_dtype)
        }
    }

    #[test]
    fn all_non_nan_binds_to_nan_count_zero() -> VortexResult<()> {
        let mut binder = TestBinder::new(true);

        let bound = bind_stats(all_non_nan(col("f")), &mut binder)?;

        assert_eq!(bound, eq(col("f_nan_count"), lit(0u64)));
        Ok(())
    }

    #[test]
    fn all_non_nan_lowers_to_null_when_nan_count_is_missing() -> VortexResult<()> {
        let mut binder = TestBinder::new(false);

        let bound = bind_stats(all_non_nan(col("f")), &mut binder)?;

        assert_eq!(bound, lit(Scalar::null(DType::Bool(Nullability::Nullable))));
        Ok(())
    }

    #[test]
    fn missing_stats_fold_when_kleene_semantics_allow_it() -> VortexResult<()> {
        let mut binder = TestBinder::new(false);

        let bound = bind_stats(and(lit(false), all_non_nan(col("f"))), &mut binder)?;

        assert_eq!(bound, lit(false));

        let bound = bind_stats(or(lit(true), all_non_nan(col("f"))), &mut binder)?;

        assert_eq!(bound, lit(true));
        Ok(())
    }

    #[test]
    fn default_binder_does_not_derive_all_non_nan_from_nan_count() -> VortexResult<()> {
        struct DefaultBinder(TestBinder);

        impl StatBinder for DefaultBinder {
            fn scope(&self) -> &DType {
                self.0.scope()
            }

            fn bound_scope(&self) -> DType {
                self.0.bound_scope()
            }

            fn bind_stat(
                &mut self,
                input: &Expression,
                stat: Stat,
                stat_dtype: &DType,
            ) -> VortexResult<Option<Expression>> {
                self.0.bind_stat(input, stat, stat_dtype)
            }
        }

        let mut binder = DefaultBinder(TestBinder::new(true));

        let bound = bind_stats(all_non_nan(col("f")), &mut binder)?;

        assert_eq!(bound, lit(Scalar::null(DType::Bool(Nullability::Nullable))));
        Ok(())
    }

    #[test]
    fn unrelated_expressions_do_not_request_nan_count() -> VortexResult<()> {
        let mut binder = TestBinder::new(false);

        let bound = bind_stats(is_null(col("f")), &mut binder)?;

        assert_eq!(bound, is_null(col("f")));
        Ok(())
    }
}
