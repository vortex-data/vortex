// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bind abstract `vortex.stat` expressions to a concrete stats representation.
//!
//! Stats rewrite rules describe pruning in terms of `vortex.stat(input, aggregate_fn)` placeholders
//! so the rewrite is independent of where statistics are stored. These stat placeholders are
//! abstract because they name the statistic needed for a proof, but not how that statistic is
//! represented by a specific layout or reader.
//!
//! Binding is the later pass that replaces each abstract placeholder with the representation used
//! by a caller: zone-map field references, file-level stat literals, or typed nulls for missing
//! stats. This lets all callers share the same falsification rules while keeping layout-specific
//! stat storage behind [`StatBinder`].

use vortex_error::VortexResult;

use crate::aggregate_fn::AggregateFnRef;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::lit;
use crate::expr::stats::Stat;
use crate::expr::traversal::NodeExt;
use crate::expr::traversal::Transformed;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::stat::StatFn;

/// A target that can bind abstract statistics to concrete expressions.
///
/// Implementations define how a pruning proof should read stats from a specific backing
/// representation. For example, a zone-map binder can translate a `max(col)` placeholder into a
/// field reference in the per-zone stats table, while a file-stats binder can translate the same
/// placeholder into a literal value from the file footer.
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
        &self,
        input: &Expression,
        stat: Stat,
        stat_dtype: &DType,
    ) -> VortexResult<Option<Expression>>;

    /// Bind `aggregate_fn(input)` to a concrete expression.
    ///
    /// Implementations should return `Ok(None)` when the requested aggregate
    /// statistic is unavailable in their backing representation. Binders that
    /// support only direct legacy [`Stat`] slots can delegate to
    /// [`bind_direct_aggregate_stat`].
    fn bind_aggregate(
        &self,
        input: &Expression,
        aggregate_fn: &AggregateFnRef,
        stat_dtype: &DType,
    ) -> VortexResult<Option<Expression>>;

    /// Expression to use when a stat is unavailable.
    ///
    /// The default is a nullable null literal, which preserves three-valued
    /// pruning semantics for stats-table execution.
    fn missing_stat(&self, dtype: DType) -> VortexResult<Expression> {
        Ok(null_expr(dtype))
    }
}

/// Bind all `vortex.stat` expressions in `predicate`.
///
/// The predicate is usually the output of a stats rewrite rule. Rewrite rules
/// are responsible for expressing stat semantics; binding maps aggregate-backed
/// stat requests to the concrete stats representation supported by the binder.
pub fn bind_stats<B: StatBinder + ?Sized>(
    predicate: Expression,
    binder: &B,
) -> VortexResult<Expression> {
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

/// Bind an aggregate function that has a direct legacy [`Stat`] slot.
pub fn bind_direct_aggregate_stat<B: StatBinder + ?Sized>(
    binder: &B,
    input: &Expression,
    aggregate_fn: &AggregateFnRef,
    stat_dtype: &DType,
) -> VortexResult<Option<Expression>> {
    let Some(stat) = Stat::from_aggregate_fn(aggregate_fn) else {
        return Ok(None);
    };
    binder.bind_stat(input, stat, stat_dtype)
}

fn bind_stat_fn(
    expr: &Expression,
    scope: &DType,
    binder: &(impl StatBinder + ?Sized),
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
    use crate::stats::nan_count;

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
            &self,
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
            &self,
            input: &Expression,
            aggregate_fn: &AggregateFnRef,
            stat_dtype: &DType,
        ) -> VortexResult<Option<Expression>> {
            bind_direct_aggregate_stat(self, input, aggregate_fn, stat_dtype)
        }
    }

    #[test]
    fn nan_count_binds_to_direct_stat_slot() -> VortexResult<()> {
        let binder = TestBinder::new(true);

        let bound = bind_stats(nan_count(col("f")), &binder)?;

        assert_eq!(bound, col("f_nan_count"));
        Ok(())
    }

    #[test]
    fn all_non_nan_does_not_derive_from_nan_count() -> VortexResult<()> {
        let binder = TestBinder::new(true);

        let bound = bind_stats(all_non_nan(col("f")), &binder)?;

        assert_eq!(bound, lit(Scalar::null(DType::Bool(Nullability::Nullable))));
        Ok(())
    }

    #[test]
    fn missing_stats_fold_when_kleene_semantics_allow_it() -> VortexResult<()> {
        let binder = TestBinder::new(false);

        let bound = bind_stats(and(lit(false), all_non_nan(col("f"))), &binder)?;

        assert_eq!(bound, lit(false));

        let bound = bind_stats(or(lit(true), all_non_nan(col("f"))), &binder)?;

        assert_eq!(bound, lit(true));
        Ok(())
    }

    #[test]
    fn unrelated_expressions_do_not_request_nan_count() -> VortexResult<()> {
        let binder = TestBinder::new(false);

        let bound = bind_stats(is_null(col("f")), &binder)?;

        assert_eq!(bound, is_null(col("f")));
        Ok(())
    }
}
