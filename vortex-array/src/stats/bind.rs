// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bind abstract `vortex.stat` expressions to a concrete stats representation.

use vortex_error::VortexResult;

use crate::aggregate_fn::AggregateFnRef;
use crate::dtype::DType;
use crate::expr::Expression;
use crate::expr::lit;
use crate::expr::stats::Stat;
use crate::scalar::Scalar;
use crate::scalar_fn::fns::binary::Binary;
use crate::scalar_fn::fns::operators::Operator;
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

    /// Bind a proof branch, rolling back any binder-local bookkeeping when the
    /// branch cannot be bound.
    ///
    /// Binders that only substitute expressions can use the default
    /// implementation. Binders that track required stats should override this
    /// so discarded proof branches do not leak requirements.
    fn bind_branch<F>(&mut self, bind: F) -> VortexResult<Option<Expression>>
    where
        Self: Sized,
        F: FnOnce(&mut Self) -> VortexResult<Option<Expression>>,
    {
        bind(self)
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
    bind_stats_expr(predicate, &scope, binder)
}

fn bind_stats_expr(
    expr: Expression,
    scope: &DType,
    binder: &mut impl StatBinder,
) -> VortexResult<Option<Expression>> {
    if expr.is::<StatFn>() {
        return match bind_stat_fn(&expr, scope, binder)? {
            Some(bound) => Ok(Some(bound)),
            None => {
                let dtype = expr.return_dtype(scope)?;
                binder.missing_stat(dtype)
            }
        };
    }

    if expr.is::<Binary>() {
        return bind_binary_expr(expr, scope, binder);
    }

    let mut children = Vec::with_capacity(expr.children().len());
    for child in expr.children().iter() {
        let Some(child) = bind_stats_expr(child.clone(), scope, binder)? else {
            return Ok(None);
        };
        children.push(child);
    }

    Ok(Some(expr.with_children(children)?))
}

fn bind_binary_expr(
    expr: Expression,
    scope: &DType,
    binder: &mut impl StatBinder,
) -> VortexResult<Option<Expression>> {
    let operator = expr.as_::<Binary>();

    match operator {
        Operator::Or => {
            let lhs = binder
                .bind_branch(|binder| bind_stats_expr(expr.child(0).clone(), scope, binder))?;
            let rhs = binder
                .bind_branch(|binder| bind_stats_expr(expr.child(1).clone(), scope, binder))?;
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => Ok(Some(expr.with_children([lhs, rhs])?)),
                (Some(expr), None) | (None, Some(expr)) => Ok(Some(expr)),
                (None, None) => Ok(None),
            }
        }
        Operator::And => binder.bind_branch(|binder| {
            let lhs = bind_stats_expr(expr.child(0).clone(), scope, binder)?;
            let rhs = bind_stats_expr(expr.child(1).clone(), scope, binder)?;
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => Ok(Some(expr.with_children([lhs, rhs])?)),
                _ => Ok(None),
            }
        }),
        _ => binder.bind_branch(|binder| {
            let lhs = bind_stats_expr(expr.child(0).clone(), scope, binder)?;
            let rhs = bind_stats_expr(expr.child(1).clone(), scope, binder)?;
            match (lhs, rhs) {
                (Some(lhs), Some(rhs)) => Ok(Some(expr.with_children([lhs, rhs])?)),
                _ => Ok(None),
            }
        }),
    }
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
