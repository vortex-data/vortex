// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Building blocks for [`StatsRewriteRule`](crate::stats::rewrite::StatsRewriteRule)
//! implementations: stat references over bound expressions and the NaN guards that keep
//! min/max comparisons sound for floating-point values.

use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::AggregateFnVTableExt;
use vortex_array::aggregate_fn::EmptyOptions as AggregateEmptyOptions;
use vortex_array::aggregate_fn::fns::all_nan::AllNan;
use vortex_array::aggregate_fn::fns::all_non_nan::AllNonNan;
use vortex_array::aggregate_fn::fns::all_non_null::AllNonNull;
use vortex_array::aggregate_fn::fns::all_null::AllNull;
use vortex_array::dtype::DType;
use vortex_array::expr::BoundExpression;
use vortex_array::expr::bound::and;
use vortex_array::expr::bound::and_collect;
use vortex_array::expr::bound::cast;
use vortex_array::expr::bound::eq;
use vortex_array::expr::bound::lit;
use vortex_array::expr::stats::Stat;
use vortex_array::scalar_fn::EmptyOptions;
use vortex_array::scalar_fn::ScalarFnVTableExt;
use vortex_array::scalar_fn::builtin::cast::Cast;
use vortex_array::scalar_fn::builtin::literal::Literal;
use vortex_array::scalar_fn::internal::row_count::RowCount;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::stats::bound::stat;

pub fn row_count() -> BoundExpression {
    RowCount
        .try_new_bound_expr(EmptyOptions, [])
        .vortex_expect("row-count expressions are always well-typed")
}

pub fn min(expr: &BoundExpression) -> Option<BoundExpression> {
    stat_expr(expr, Stat::Min)
}

pub fn max(expr: &BoundExpression) -> Option<BoundExpression> {
    stat_expr(expr, Stat::Max)
}

pub fn null_count(expr: &BoundExpression) -> Option<BoundExpression> {
    stat_expr(expr, Stat::NullCount)
}

pub fn nan_count(expr: &BoundExpression) -> Option<BoundExpression> {
    stat_expr(expr, Stat::NaNCount)
}

pub fn all_null(expr: &BoundExpression) -> BoundExpression {
    stat_fn(expr.clone(), AllNull.bind(AggregateEmptyOptions))
}

pub fn all_non_null(expr: &BoundExpression) -> BoundExpression {
    stat_fn(expr.clone(), AllNonNull.bind(AggregateEmptyOptions))
}

pub fn all_nan(expr: &BoundExpression) -> BoundExpression {
    stat_fn(expr.clone(), AllNan.bind(AggregateEmptyOptions))
}

pub fn all_non_nan(expr: &BoundExpression) -> BoundExpression {
    stat_fn(expr.clone(), AllNonNan.bind(AggregateEmptyOptions))
}

pub enum NanCheck {
    NotNeeded,
    Check(BoundExpression),
    Unavailable,
}

pub trait NonNanProof {
    const EMIT_UNGUARDED_REWRITES: bool;

    fn check(expr: &BoundExpression) -> VortexResult<NanCheck>;
}

pub struct NanCountProof;

impl NonNanProof for NanCountProof {
    const EMIT_UNGUARDED_REWRITES: bool = true;

    fn check(expr: &BoundExpression) -> VortexResult<NanCheck> {
        non_nan_check(expr, |expr| match stat_expr(expr, Stat::NaNCount) {
            Some(nan_count) => NanCheck::Check(eq(nan_count, lit(0u64))),
            None => NanCheck::Unavailable,
        })
    }
}

pub struct AllNonNanProof;

impl NonNanProof for AllNonNanProof {
    const EMIT_UNGUARDED_REWRITES: bool = false;

    fn check(expr: &BoundExpression) -> VortexResult<NanCheck> {
        non_nan_check(expr, |expr| {
            NanCheck::Check(stat_fn(expr.clone(), AllNonNan.bind(AggregateEmptyOptions)))
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

pub fn stat_expr(expr: &BoundExpression, stat: Stat) -> Option<BoundExpression> {
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

    let aggregate_fn = stat.aggregate_fn()?;
    // The aggregate may not support the expression's dtype, e.g. min/max over structs,
    // even when the predicate itself is well-typed. Such stats cannot be lowered later,
    // so do not reference them in the rewrite.
    aggregate_fn
        .return_dtype(expr.dtype())
        .is_some()
        .then(|| stat_fn(expr.clone(), aggregate_fn))
}

pub fn with_non_nan_guards<'a, P: NonNanProof>(
    exprs: impl IntoIterator<Item = &'a BoundExpression>,
    value_predicate: BoundExpression,
) -> VortexResult<Option<BoundExpression>> {
    let mut nan_checks = Vec::new();
    for expr in exprs {
        match P::check(expr)? {
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

pub fn literal_stat(expr: &BoundExpression, stat: Stat) -> Option<BoundExpression> {
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

pub fn cast_stat(expr: &BoundExpression, dtype: &DType, stat: Stat) -> Option<BoundExpression> {
    match stat {
        Stat::Min | Stat::Max => stat_expr(expr, stat).map(|stat| cast(stat, dtype.clone())),
        Stat::NaNCount | Stat::Sum | Stat::UncompressedSizeInBytes => stat_expr(expr, stat),
        Stat::NullCount | Stat::IsConstant | Stat::IsSorted | Stat::IsStrictSorted => None,
    }
}

pub fn stat_fn(expr: BoundExpression, aggregate_fn: AggregateFnRef) -> BoundExpression {
    stat(expr, aggregate_fn)
}
