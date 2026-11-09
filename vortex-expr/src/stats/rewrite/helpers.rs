// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Building blocks for [`StatsRewriteRule`](crate::stats::rewrite::StatsRewriteRule)
//! implementations: stat references over bound expressions and the NaN guards that keep
//! min/max comparisons sound for floating-point values.

use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions as AggregateEmptyOptions;
use vortex_array::aggregate_fn::NumericalAggregateOpts;
use vortex_array::aggregate_fn::fns::all_nan::AllNan;
use vortex_array::aggregate_fn::fns::all_non_nan::AllNonNan;
use vortex_array::aggregate_fn::fns::all_non_null::AllNonNull;
use vortex_array::aggregate_fn::fns::all_null::AllNull;
use vortex_array::aggregate_fn::fns::max::Max;
use vortex_array::aggregate_fn::fns::min::Min;
use vortex_array::aggregate_fn::fns::nan_count::NanCount;
use vortex_array::aggregate_fn::fns::null_count::NullCount;
use vortex_array::dtype::DType;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::builtin::cast::Cast;
use vortex_array::scalar_fn::builtin::literal::Literal;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::BoundExpression;
use crate::ScalarFnExprExt;
use crate::bound::and;
use crate::bound::and_collect;
use crate::bound::eq;
use crate::bound::lit;
use crate::lambda::row_count::RowCount;
use crate::stats::derive::stat_expr;

pub fn row_count() -> BoundExpression {
    RowCount
        .try_new_bound_expr(EmptyOptions, [])
        .vortex_expect("row-count expressions are always well-typed")
}

// Statistics follow NaN-skipping semantics, so min and max request it explicitly.
pub fn min(expr: &BoundExpression, session: &VortexSession) -> Option<BoundExpression> {
    stat_expr(
        expr,
        &Min.bind(NumericalAggregateOpts::skip_nans()),
        session,
    )
}

pub fn max(expr: &BoundExpression, session: &VortexSession) -> Option<BoundExpression> {
    stat_expr(
        expr,
        &Max.bind(NumericalAggregateOpts::skip_nans()),
        session,
    )
}

pub fn null_count(expr: &BoundExpression, session: &VortexSession) -> Option<BoundExpression> {
    stat_expr(expr, &NullCount.bind(AggregateEmptyOptions), session)
}

pub fn nan_count(expr: &BoundExpression, session: &VortexSession) -> Option<BoundExpression> {
    stat_expr(expr, &NanCount.bind(AggregateEmptyOptions), session)
}

pub fn all_null(expr: &BoundExpression, session: &VortexSession) -> Option<BoundExpression> {
    stat_expr(expr, &AllNull.bind(AggregateEmptyOptions), session)
}

pub fn all_non_null(expr: &BoundExpression, session: &VortexSession) -> Option<BoundExpression> {
    stat_expr(expr, &AllNonNull.bind(AggregateEmptyOptions), session)
}

pub fn all_nan(expr: &BoundExpression, session: &VortexSession) -> Option<BoundExpression> {
    stat_expr(expr, &AllNan.bind(AggregateEmptyOptions), session)
}

pub fn all_non_nan(expr: &BoundExpression, session: &VortexSession) -> Option<BoundExpression> {
    stat_expr(expr, &AllNonNan.bind(AggregateEmptyOptions), session)
}

pub enum NanCheck {
    NotNeeded,
    Check(BoundExpression),
    Unavailable,
}

pub trait NonNanProof {
    const EMIT_UNGUARDED_REWRITES: bool;

    fn check(expr: &BoundExpression, session: &VortexSession) -> VortexResult<NanCheck>;
}

pub struct NanCountProof;

impl NonNanProof for NanCountProof {
    const EMIT_UNGUARDED_REWRITES: bool = true;

    fn check(expr: &BoundExpression, session: &VortexSession) -> VortexResult<NanCheck> {
        non_nan_check(expr, |expr| match nan_count(expr, session) {
            Some(nan_count) => NanCheck::Check(eq(nan_count, lit(0u64))),
            None => NanCheck::Unavailable,
        })
    }
}

pub struct AllNonNanProof;

impl NonNanProof for AllNonNanProof {
    const EMIT_UNGUARDED_REWRITES: bool = false;

    fn check(expr: &BoundExpression, session: &VortexSession) -> VortexResult<NanCheck> {
        non_nan_check(expr, |expr| match all_non_nan(expr, session) {
            Some(all_non_nan) => NanCheck::Check(all_non_nan),
            None => NanCheck::Unavailable,
        })
    }
}

// Min/max do not order NaN values, so comparison rewrites are only sound when every
// candidate value is known to be non-NaN. Cast result dtypes are not enough: a cast
// from float to non-float still needs a proof about the float source values.
pub fn non_nan_check(
    expr: &BoundExpression,
    proof: impl FnOnce(&BoundExpression) -> NanCheck,
) -> VortexResult<NanCheck> {
    if let Some(scalar) = expr.as_opt::<Literal>() {
        if !scalar.dtype().is_float() {
            return Ok(NanCheck::NotNeeded);
        }
        return Ok(if scalar.as_primitive().is_nan() {
            NanCheck::Check(lit(false))
        } else {
            NanCheck::NotNeeded
        });
    }

    if expr.is::<Cast>() {
        if !has_nans(expr.child(0).dtype()) {
            return Ok(NanCheck::NotNeeded);
        }

        return non_nan_check(expr.child(0), proof);
    }

    if !has_nans(expr.dtype()) {
        return Ok(NanCheck::NotNeeded);
    }

    Ok(proof(expr))
}

fn has_nans(dtype: &DType) -> bool {
    dtype.is_float()
}

pub fn with_non_nan_guards<'a, P: NonNanProof>(
    exprs: impl IntoIterator<Item = &'a BoundExpression>,
    value_predicate: BoundExpression,
    session: &VortexSession,
) -> VortexResult<Option<BoundExpression>> {
    let mut nan_checks = Vec::new();
    for expr in exprs {
        match P::check(expr, session)? {
            NanCheck::NotNeeded => {}
            NanCheck::Check(check) => nan_checks.push(check),
            NanCheck::Unavailable => return Ok(None),
        }
    }
    let nan_predicate = and_collect(nan_checks);

    Ok(match nan_predicate {
        Some(nan_check) => Some(and(nan_check, value_predicate)),
        // No possible NaN-bearing expression remains, so the value predicate is
        // already guarded. Only one registered rule emits this unguarded
        // rewrite so non-float comparisons are not duplicated.
        None if P::EMIT_UNGUARDED_REWRITES => Some(value_predicate),
        None => None,
    })
}
