// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Several statistics of an integer array in one pass.

use num_traits::AsPrimitive;
use num_traits::Bounded;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use super::IsConstant;
use super::IsSorted;
use super::MinMax;
use super::Sum;
use super::accumulate;
use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::arrays::Primitive;
use crate::dtype::NativePType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::match_each_integer_ptype;
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

/// Computes the requested statistics of an integer array that are not cached yet, in one pass
/// over its values, and caches them.
///
/// Does nothing for other arrays, or when at most one statistic would be computed, which the
/// statistic's own aggregate computes as fast.
pub(crate) fn compute_primitive_stats(
    array: &ArrayRef,
    stats: &[Stat],
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let Some(primitive) = array.as_opt::<Primitive>() else {
        return Ok(());
    };
    // Arrays of at most one value have their own rules, and nothing to gain.
    if !primitive.ptype().is_int() || array.len() < 2 {
        return Ok(());
    }

    let cache = array.statistics();
    let want = |stat: Stat| stats.contains(&stat) && !cache.get(stat).is_exact();
    let wanted = Wanted {
        min_max: want(Stat::Min) || want(Stat::Max),
        sum: want(Stat::Sum),
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
    match_each_integer_ptype!(primitive.ptype(), |T| {
        compute::<T>(primitive.as_slice::<T>(), &validity, wanted, &cache)
    })?;
    cache.set(
        Stat::NullCount,
        Precision::Exact(ScalarValue::from(validity.false_count())),
    );
    Ok(())
}

/// Computes the `wanted` statistics of `values` in one pass, and caches them.
fn compute<T>(
    values: &[T],
    validity: &Mask,
    wanted: Wanted,
    cache: &StatsSetRef<'_>,
) -> VortexResult<()>
where
    T: NativePType + Ord + Bounded + AsPrimitive<i64> + AsPrimitive<u64>,
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

    if let Some((min, max)) = min_max.and_then(|acc| acc.finish()) {
        cache.set(Stat::Min, Precision::Exact(PValue::from(min).into()));
        cache.set(Stat::Max, Precision::Exact(PValue::from(max).into()));
    }
    if let Some(sum) = sum.and_then(|acc| acc.finish()) {
        let sum = if T::PTYPE.is_signed_int() {
            ScalarValue::from(i64::try_from(sum)?)
        } else {
            ScalarValue::from(u64::try_from(sum)?)
        };
        cache.set(Stat::Sum, Precision::Exact(sum));
    }
    if let Some(constant) = constant.and_then(|acc| acc.finish()) {
        cache.set(Stat::IsConstant, Precision::Exact(constant.into()));
    }
    if let Some(acc) = sorted {
        cache.set(Stat::IsSorted, Precision::Exact(acc.finish().into()));
    }
    if let Some(acc) = strict_sorted {
        cache.set(Stat::IsStrictSorted, Precision::Exact(acc.finish().into()));
    }
    Ok(())
}
