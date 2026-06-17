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
        if aggregate_fn.is::<AllNan>() {
            let Some(nan_count) = self.bind_legacy_stat(input, Stat::NaNCount)? else {
                return Ok(None);
            };
            return Ok(Some(eq(nan_count, RowCount.new_expr(EmptyOptions, []))));
        }

        if aggregate_fn.is::<AllNonNan>() {
            let Some(nan_count) = self.bind_legacy_stat(input, Stat::NaNCount)? else {
                return Ok(None);
            };
            return Ok(Some(eq(nan_count, lit(0u64))));
        }

        if aggregate_fn.is::<AllNull>() {
            let Some(null_count) = self.bind_legacy_stat(input, Stat::NullCount)? else {
                return Ok(None);
            };
            return Ok(Some(eq(null_count, RowCount.new_expr(EmptyOptions, []))));
        }

        if aggregate_fn.is::<AllNonNull>() {
            let Some(null_count) = self.bind_legacy_stat(input, Stat::NullCount)? else {
                return Ok(None);
            };
            return Ok(Some(eq(null_count, lit(0u64))));
        }

        let Some(stat) = Stat::from_aggregate_fn(aggregate_fn) else {
            return Ok(None);
        };
        self.bind_stat(input, stat, stat_dtype)
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
