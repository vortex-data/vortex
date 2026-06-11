// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bind abstract `vortex.stat` expressions to a concrete stats representation.

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
pub trait StatBinder {
    /// The dtype scope used to type-check expressions before stats are bound.
    fn scope(&self) -> &DType;

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
        let Some(stat) = Stat::from_aggregate_fn(aggregate_fn) else {
            return Ok(None);
        };
        self.bind_stat(input, stat, stat_dtype)
    }

    /// Expression to use when a stat is unavailable.
    ///
    /// The default is a nullable null literal, which preserves three-valued
    /// pruning semantics for stats-table execution. Catalog-like binders can
    /// return `Ok(None)` to reject expressions that require unavailable stats.
    fn missing_stat(&mut self, dtype: DType) -> VortexResult<Option<Expression>> {
        Ok(Some(null_expr(dtype)))
    }
}

/// Bind all `vortex.stat` expressions in `predicate`.
///
/// The predicate is usually the output of a stats rewrite rule. Rewrite rules
/// are responsible for expressing stat semantics; binding maps aggregate-backed
/// stat requests to the concrete stats representation supported by the binder.
pub fn bind_stats(
    predicate: Expression,
    binder: &mut impl StatBinder,
) -> VortexResult<Option<Expression>> {
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
                    match binder.missing_stat(dtype.clone())? {
                        Some(missing) => Ok(Transformed::yes(missing)),
                        None => Ok(Transformed::yes(null_expr(dtype))),
                    }
                }
            }
        })?
        .into_inner();

    #[expect(deprecated)]
    let lowered = lowered.simplify_untyped()?;
    Ok(Some(lowered))
}

fn bind_stat_fn(
    expr: &Expression,
    scope: &DType,
    binder: &mut impl StatBinder,
) -> VortexResult<Option<Expression>> {
    let options = expr.as_::<StatFn>();
    let aggregate_fn = options.aggregate_fn();
    let input = expr.child(0);

    let stat_dtype = expr.return_dtype(scope)?;
    binder.bind_aggregate(input, aggregate_fn, &stat_dtype)
}

fn null_expr(dtype: DType) -> Expression {
    lit(Scalar::null(dtype.as_nullable()))
}
