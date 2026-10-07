// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Several statistics of a primitive or decimal array in one pass.

use num_traits::AsPrimitive;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::Extremes;
use super::FloatKey;
use super::FloatSum;
use super::IsConstant;
use super::IsSorted;
use super::Keyed;
use super::MinMax;
use super::Sum;
use super::accumulate;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::Decimal;
use crate::arrays::Primitive;
use crate::dtype::NativeDecimalType;
use crate::dtype::NativePType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::match_each_decimal_value_type;
use crate::match_each_float_ptype;
use crate::match_each_integer_ptype;
use crate::scalar::DecimalValue;
use crate::scalar::PValue;
use crate::scalar::ScalarValue;
use crate::stats::StatsSetRef;

/// The statistics to compute in one pass.
#[derive(Clone, Copy)]
struct Wanted {
    min_max: bool,
    sum: bool,
    constant: bool,
    sorted: bool,
    strict_sorted: bool,
}

/// Computes the requested statistics of a primitive or decimal array that are not cached yet, in
/// one pass over its values, and caches them.
///
/// Does nothing for other arrays, or when at most one statistic would be computed, which the
/// statistic's own aggregate computes as fast.
pub(crate) fn compute_primitive_stats(
    array: &ArrayRef,
    stats: &[Stat],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    // Arrays of at most one value have their own rules, and nothing to gain. Neither does a single
    // statistic, which is checked first because it is the common request and costs no lookup.
    let fused = [
        Stat::Min,
        Stat::Max,
        Stat::Sum,
        Stat::IsConstant,
        Stat::IsSorted,
        Stat::IsStrictSorted,
    ];
    if array.len() < 2 || stats.iter().filter(|s| fused.contains(s)).count() < 2 {
        return Ok(());
    }
    let primitive = array.as_opt::<Primitive>();
    let decimal = array.as_opt::<Decimal>();
    if primitive.is_none() && decimal.is_none() {
        return Ok(());
    }

    let cache = array.statistics();
    let want = |stat: Stat| stats.contains(&stat) && !cache.get(stat).is_exact();
    let wanted = Wanted {
        min_max: want(Stat::Min) || want(Stat::Max),
        // The sum of decimals widens their precision, which the sum aggregate decides.
        sum: primitive.is_some() && want(Stat::Sum),
        constant: want(Stat::IsConstant),
        sorted: want(Stat::IsSorted),
        strict_sorted: want(Stat::IsStrictSorted),
    };
    let count = [
        wanted.min_max,
        wanted.sum,
        wanted.constant,
        wanted.sorted,
        wanted.strict_sorted,
    ]
    .into_iter()
    .filter(|&w| w)
    .count();
    if count < 2 {
        return Ok(());
    }

    let validity = array.validity()?.execute_mask(array.len(), ctx)?;
    if let Some(primitive) = primitive {
        let ptype = primitive.ptype();
        if ptype.is_int() {
            match_each_integer_ptype!(ptype, |T| {
                integers::<T>(primitive.as_slice::<T>(), &validity, wanted, &cache)
            })?;
        } else {
            match_each_float_ptype!(ptype, |F| {
                floats::<F>(primitive.as_slice::<F>(), &validity, wanted, &cache);
            });
        }
    } else if let Some(decimal) = decimal {
        match_each_decimal_value_type!(decimal.values_type(), |D| {
            decimals::<D>(&decimal.buffer::<D>(), &validity, wanted, &cache);
        });
    }
    cache.set(
        Stat::NullCount,
        Precision::Exact(ScalarValue::from(validity.false_count())),
    );
    Ok(())
}

/// Caches the order statistics, which have the same accumulators for every type.
fn set_orders(
    cache: &StatsSetRef<'_>,
    constant: Option<Option<bool>>,
    sorted: Option<bool>,
    strict_sorted: Option<bool>,
) {
    if let Some(Some(constant)) = constant {
        cache.set(Stat::IsConstant, Precision::Exact(constant.into()));
    }
    if let Some(sorted) = sorted {
        cache.set(Stat::IsSorted, Precision::Exact(sorted.into()));
    }
    if let Some(strict_sorted) = strict_sorted {
        cache.set(Stat::IsStrictSorted, Precision::Exact(strict_sorted.into()));
    }
}

/// Caches the bounds.
fn set_bounds(cache: &StatsSetRef<'_>, bounds: Option<(ScalarValue, ScalarValue)>) {
    if let Some((min, max)) = bounds {
        cache.set(Stat::Min, Precision::Exact(min));
        cache.set(Stat::Max, Precision::Exact(max));
    }
}

/// Computes the `wanted` statistics of integers in one pass, and caches them.
fn integers<T>(
    values: &[T],
    validity: &Mask,
    wanted: Wanted,
    cache: &StatsSetRef<'_>,
) -> VortexResult<()>
where
    T: NativePType + Extremes + AsPrimitive<i64> + AsPrimitive<u64>,
    PValue: From<T>,
{
    let mut acc = (
        wanted.min_max.then(MinMax::<T>::new),
        wanted.sum.then(Sum::<T>::new),
        wanted.constant.then(IsConstant::<T>::new),
        wanted.sorted.then(IsSorted::<T, false>::new),
        wanted.strict_sorted.then(IsSorted::<T, true>::new),
    );
    accumulate(values, validity, &mut acc);
    let (min_max, sum, constant, sorted, strict_sorted) = acc;

    set_bounds(
        cache,
        min_max
            .and_then(|acc| acc.finish())
            .map(|(min, max)| (PValue::from(min).into(), PValue::from(max).into())),
    );
    if let Some(sum) = sum.and_then(|acc| acc.finish()) {
        let sum = if T::PTYPE.is_signed_int() {
            ScalarValue::from(i64::try_from(sum)?)
        } else {
            ScalarValue::from(u64::try_from(sum)?)
        };
        cache.set(Stat::Sum, Precision::Exact(sum));
    }
    set_orders(
        cache,
        constant.map(|acc| acc.finish()),
        sorted.map(|acc| acc.finish()),
        strict_sorted.map(|acc| acc.finish()),
    );
    Ok(())
}

/// Computes the `wanted` statistics of floats in one pass, and caches them. The bounds and the sum
/// skip NaNs, as the statistics do.
fn floats<F>(values: &[F], validity: &Mask, wanted: Wanted, cache: &StatsSetRef<'_>)
where
    F: FloatKey,
    PValue: From<F>,
{
    let mut acc = (
        wanted
            .min_max
            .then(|| Keyed::<F, _, true>::new(MinMax::<F::Key>::new())),
        wanted.sum.then(FloatSum::<F, true>::new),
        wanted
            .constant
            .then(|| Keyed::<F, _, false>::new(IsConstant::<F::Key>::new())),
        wanted
            .sorted
            .then(|| Keyed::<F, _, false>::new(IsSorted::<F::Key, false>::new())),
        wanted
            .strict_sorted
            .then(|| Keyed::<F, _, false>::new(IsSorted::<F::Key, true>::new())),
    );
    accumulate(values, validity, &mut acc);
    let (min_max, sum, constant, sorted, strict_sorted) = acc;

    set_bounds(
        cache,
        min_max
            .and_then(|acc| acc.inner().finish())
            .map(|(min, max)| {
                (
                    PValue::from(F::from_key(min)).into(),
                    PValue::from(F::from_key(max)).into(),
                )
            }),
    );
    if let Some(sum) = sum {
        cache.set(Stat::Sum, Precision::Exact(ScalarValue::from(sum.finish())));
    }
    set_orders(
        cache,
        constant.map(|acc| acc.inner().finish()),
        sorted.map(|acc| acc.inner().finish()),
        strict_sorted.map(|acc| acc.inner().finish()),
    );
}

/// Computes the `wanted` statistics of decimals, by their storage integers, in one pass, and caches
/// them.
fn decimals<D>(values: &[D], validity: &Mask, wanted: Wanted, cache: &StatsSetRef<'_>)
where
    D: NativeDecimalType + Extremes + Into<DecimalValue>,
{
    let mut acc = (
        wanted.min_max.then(MinMax::<D>::new),
        wanted.constant.then(IsConstant::<D>::new),
        wanted.sorted.then(IsSorted::<D, false>::new),
        wanted.strict_sorted.then(IsSorted::<D, true>::new),
    );
    accumulate(values, validity, &mut acc);
    let (min_max, constant, sorted, strict_sorted) = acc;

    set_bounds(
        cache,
        min_max
            .and_then(|acc| acc.finish())
            .map(|(min, max)| (ScalarValue::from(min.into()), ScalarValue::from(max.into()))),
    );
    set_orders(
        cache,
        constant.map(|acc| acc.finish()),
        sorted.map(|acc| acc.finish()),
        strict_sorted.map(|acc| acc.finish()),
    );
}
