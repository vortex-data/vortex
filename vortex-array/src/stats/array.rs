// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Temporary fixed-statistics access during the aggregate cache migration.
//!
//! The facade converts historical scalar types at its boundary. Detached projections support the
//! remaining legacy consumers without retaining another mutable store.

use enum_iterator::all;
use vortex_array::ExecutionCtx;
use vortex_error::VortexError;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use super::AggregationsRef;
use super::StatsSet;
use super::StatsSetIntoIter;
use super::TypedStatsSetRef;
use crate::ArrayRef;
use crate::aggregate_fn::NumericalAggregateOpts;
use crate::aggregate_fn::fns::is_constant::is_constant;
use crate::aggregate_fn::fns::is_sorted::is_sorted;
use crate::aggregate_fn::fns::is_sorted::is_strict_sorted;
use crate::aggregate_fn::fns::min_max::MinMaxResult;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::aggregate_fn::fns::nan_count::nan_count;
use crate::aggregate_fn::fns::sum::sum;
use crate::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use crate::dtype::DType;
use crate::dtype::PType;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// Temporary fixed-statistics view over the array's finalized aggregate cache.
///
/// Legacy projections cannot represent exact nulls or custom aggregate functions.
pub struct StatsSetRef<'a> {
    dyn_array_ref: &'a ArrayRef,
    aggregations: AggregationsRef<'a>,
}

impl<'a> StatsSetRef<'a> {
    pub(crate) fn new(array: &'a ArrayRef, aggregations: AggregationsRef<'a>) -> Self {
        Self {
            dyn_array_ref: array,
            aggregations,
        }
    }
}

impl StatsSetRef<'_> {
    pub(crate) fn replace(&self, stats: StatsSet) {
        // Replace representable legacy fields without discarding generic facts or exact nulls.
        for stat in all::<Stat>() {
            if !self.get(stat).is_absent() {
                self.clear(stat);
            }
        }
        self.set_iter(stats.into_iter());
    }

    pub fn set_iter(&self, iter: StatsSetIntoIter) {
        for (stat, value) in iter {
            self.set(stat, value);
        }
    }

    /// Inherit portable results after preserving logical values, validity, and order.
    pub fn inherit_from(&self, stats: StatsSetRef<'_>) {
        self.aggregations.inherit_from(stats.aggregations);
    }

    pub fn inherit<'a>(&self, iter: impl Iterator<Item = &'a (Stat, Precision<ScalarValue>)>) {
        for (stat, value) in iter {
            if value.is_exact() || !self.get(*stat).is_exact() {
                self.set(*stat, value.clone());
            }
        }
    }

    pub fn with_typed_stats_set<U, F: FnOnce(TypedStatsSetRef) -> U>(&self, apply: F) -> U {
        let stats = self.to_owned();
        apply(stats.as_typed_ref(self.dyn_array_ref.dtype()))
    }

    pub fn to_owned(&self) -> StatsSet {
        all::<Stat>()
            .filter_map(|stat| {
                let value = self.get(stat).and_then(Scalar::into_value);
                (!value.is_absent()).then_some((stat, value))
            })
            .collect()
    }

    pub fn with_iter<
        F: for<'a> FnOnce(&mut dyn Iterator<Item = &'a (Stat, Precision<ScalarValue>)>) -> R,
        R,
    >(
        &self,
        f: F,
    ) -> R {
        let stats = self.to_owned();
        f(&mut stats.iter())
    }

    /// Returns the value of `stat` by either fetching it from cache if it exists and is [`Precision::Exact`], or falling back to
    /// computation. The underlying compute kernels will cache the computed stat in the latter case.
    pub fn compute_stat(&self, stat: Stat, ctx: &mut ExecutionCtx) -> VortexResult<Option<Scalar>> {
        // If it's already computed and exact, we can return it.
        if let Precision::Exact(s) = self.get(stat) {
            return Ok(Some(s));
        }

        Ok(match stat {
            Stat::Min => min_max(self.dyn_array_ref, ctx, NumericalAggregateOpts::default())?
                .map(|MinMaxResult { min, max: _ }| min),
            Stat::Max => min_max(self.dyn_array_ref, ctx, NumericalAggregateOpts::default())?
                .map(|MinMaxResult { min: _, max }| max),
            Stat::Sum => {
                Stat::Sum
                    .dtype(self.dyn_array_ref.dtype())
                    .is_some()
                    .then(|| {
                        // Sum is supported for this dtype.
                        sum(self.dyn_array_ref, ctx)
                    })
                    .transpose()?
            }
            Stat::NullCount => self.dyn_array_ref.invalid_count(ctx).ok().map(Into::into),
            Stat::IsConstant => {
                if self.dyn_array_ref.is_empty() {
                    None
                } else {
                    Some(is_constant(self.dyn_array_ref, ctx)?.into())
                }
            }
            Stat::IsSorted => Some(is_sorted(self.dyn_array_ref, ctx)?.into()),
            Stat::IsStrictSorted => Some(is_strict_sorted(self.dyn_array_ref, ctx)?.into()),
            Stat::UncompressedSizeInBytes => Stat::UncompressedSizeInBytes
                .dtype(self.dyn_array_ref.dtype())
                .is_some()
                .then(|| uncompressed_size_in_bytes(self.dyn_array_ref, ctx))
                .transpose()?
                .map(|s| s.into()),
            Stat::NaNCount => {
                Stat::NaNCount
                    .dtype(self.dyn_array_ref.dtype())
                    .is_some()
                    .then(|| {
                        // NaNCount is supported for this dtype.
                        nan_count(self.dyn_array_ref, ctx)
                    })
                    .transpose()?
                    .map(|s| s.into())
            }
        })
    }

    pub fn compute_all(&self, stats: &[Stat], ctx: &mut ExecutionCtx) -> VortexResult<StatsSet> {
        let mut stats_set = StatsSet::default();
        for &stat in stats {
            if let Some(s) = self.compute_stat(stat, ctx)?
                && let Some(value) = s.into_value()
            {
                stats_set.set(stat, Precision::exact(value));
            }
        }
        Ok(stats_set)
    }
}

impl StatsSetRef<'_> {
    pub fn compute_as<U: for<'a> TryFrom<&'a Scalar, Error = VortexError>>(
        &self,
        stat: Stat,
        ctx: &mut ExecutionCtx,
    ) -> Option<U> {
        self.compute_stat(stat, ctx)
            .inspect_err(|e| tracing::warn!("Failed to compute stat {stat}: {e}"))
            .ok()
            .flatten()
            .map(|s| U::try_from(&s))
            .transpose()
            .unwrap_or_else(|err| {
                vortex_panic!(
                    err,
                    "Failed to compute stat {} as {}",
                    stat,
                    std::any::type_name::<U>()
                )
            })
    }

    /// Replace a legacy producer fact, or clear it when absent.
    ///
    /// The caller must supply a fact that describes this input. This temporary interface preserves
    /// existing producer and representation-transfer callers during migration.
    pub fn set(&self, stat: Stat, value: Precision<ScalarValue>) {
        if value.is_absent() {
            self.clear(stat);
            return;
        }
        let dtype = self
            .legacy_dtype(stat)
            .vortex_expect("legacy statistic does not support array dtype");
        let dtype = if stat.has_same_dtype_as_array() {
            dtype.as_nullable()
        } else {
            dtype
        };
        let result = value.into_scalar(dtype);
        self.aggregations
            .set_result(stat.finalized_aggregate_fn().clone(), result);
    }

    fn legacy_dtype(&self, stat: Stat) -> Option<DType> {
        match stat {
            // Historical count fields exist even when the current aggregate declines this dtype.
            Stat::NullCount | Stat::NaNCount | Stat::UncompressedSizeInBytes => {
                Some(PType::U64.into())
            }
            _ => stat.dtype(self.dyn_array_ref.dtype()),
        }
    }

    pub fn clear(&self, stat: Stat) {
        self.aggregations
            .clear_result(stat.finalized_aggregate_fn());
    }

    pub fn compute_min<U: for<'a> TryFrom<&'a Scalar, Error = VortexError>>(
        &self,
        ctx: &mut ExecutionCtx,
    ) -> Option<U> {
        self.compute_as(Stat::Min, ctx)
    }

    pub fn compute_max<U: for<'a> TryFrom<&'a Scalar, Error = VortexError>>(
        &self,
        ctx: &mut ExecutionCtx,
    ) -> Option<U> {
        self.compute_as(Stat::Max, ctx)
    }

    pub fn compute_is_sorted(&self, ctx: &mut ExecutionCtx) -> Option<bool> {
        self.compute_as(Stat::IsSorted, ctx)
    }

    pub fn compute_is_strict_sorted(&self, ctx: &mut ExecutionCtx) -> Option<bool> {
        self.compute_as(Stat::IsStrictSorted, ctx)
    }

    pub fn compute_is_constant(&self, ctx: &mut ExecutionCtx) -> Option<bool> {
        self.compute_as(Stat::IsConstant, ctx)
    }

    pub fn compute_null_count(&self, ctx: &mut ExecutionCtx) -> Option<usize> {
        self.compute_as(Stat::NullCount, ctx)
    }

    pub fn compute_uncompressed_size_in_bytes(&self, ctx: &mut ExecutionCtx) -> Option<usize> {
        self.compute_as(Stat::UncompressedSizeInBytes, ctx)
    }
}

impl StatsProvider for StatsSetRef<'_> {
    fn get(&self, stat: Stat) -> Precision<Scalar> {
        self.aggregations
            .get_result(stat.finalized_aggregate_fn())
            .and_then(|scalar| {
                if scalar.is_null() {
                    return None;
                }
                let dtype = self.legacy_dtype(stat)?;
                Some(if scalar.dtype() == &dtype {
                    scalar
                } else {
                    scalar
                        .cast(&dtype)
                        .vortex_expect("cached legacy statistic has an incompatible dtype")
                })
            })
    }

    fn len(&self) -> usize {
        all::<Stat>()
            .filter(|stat| !self.get(*stat).is_absent())
            .count()
    }
}
