// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Bind abstract `vortex.stat` expressions to a concrete stats representation.

use vortex_error::VortexResult;

use crate::aggregate_fn::fns::all_nan::AllNan;
use crate::aggregate_fn::fns::all_non_nan::AllNonNan;
use crate::aggregate_fn::fns::all_non_null::AllNonNull;
use crate::aggregate_fn::fns::all_null::AllNull;
use crate::aggregate_fn::fns::nan_count::NanCount;
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
/// The predicate is usually the output of a stats rewrite rule. This function
/// centralizes the legacy aggregate/stat mapping: `all_null` and `all_nan`
/// style aggregate expressions are expanded through exact count stats, while
/// direct aggregate stats are delegated to the supplied binder.
pub fn bind_stats(
    predicate: Expression,
    binder: &mut impl StatBinder,
) -> VortexResult<Option<Expression>> {
    let scope = binder.scope().clone();
    let mut missing_stat = false;
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
                        None => {
                            missing_stat = true;
                            Ok(Transformed::yes(null_expr(dtype)))
                        }
                    }
                }
            }
        })?
        .into_inner();

    if missing_stat {
        return Ok(None);
    }

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
    let input_dtype = input.return_dtype(scope)?;

    if aggregate_fn.is::<AllNan>() {
        if !has_nans(&input_dtype) {
            return Ok(Some(lit(false)));
        }
        let stat_dtype = expr.return_dtype(scope)?;
        return Ok(binder
            .bind_stat(input, Stat::NaNCount, &stat_dtype)?
            .map(|stat| eq(stat, row_count_expr())));
    }

    if aggregate_fn.is::<AllNonNan>() {
        if !has_nans(&input_dtype) {
            return Ok(Some(lit(true)));
        }
        let stat_dtype = expr.return_dtype(scope)?;
        return Ok(binder
            .bind_stat(input, Stat::NaNCount, &stat_dtype)?
            .map(|stat| eq(stat, lit(0u64))));
    }

    if aggregate_fn.is::<NanCount>() && !has_nans(&input_dtype) {
        return Ok(Some(lit(0u64)));
    }

    if aggregate_fn.is::<AllNull>() {
        let stat_dtype = expr.return_dtype(scope)?;
        return Ok(binder
            .bind_stat(input, Stat::NullCount, &stat_dtype)?
            .map(|stat| eq(stat, row_count_expr())));
    }

    if aggregate_fn.is::<AllNonNull>() {
        let stat_dtype = expr.return_dtype(scope)?;
        return Ok(binder
            .bind_stat(input, Stat::NullCount, &stat_dtype)?
            .map(|stat| eq(stat, lit(0u64))));
    }

    let Some(stat) = Stat::from_aggregate_fn(aggregate_fn) else {
        return Ok(None);
    };

    let stat_dtype = expr.return_dtype(scope)?;
    binder.bind_stat(input, stat, &stat_dtype)
}

fn row_count_expr() -> Expression {
    RowCount.new_expr(EmptyOptions, [])
}

fn null_expr(dtype: DType) -> Expression {
    lit(Scalar::null(dtype.as_nullable()))
}

fn has_nans(dtype: &DType) -> bool {
    matches!(dtype, DType::Primitive(ptype, _) if ptype.is_float())
}
