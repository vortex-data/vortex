// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stats as they are stored on arrays.

use std::sync::Arc;

use parking_lot::RwLock;
use vortex_array::ExecutionCtx;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use super::MutTypedStatsSetRef;
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
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// A shared [`StatsSet`] stored in an array. Can be shared by copies of the array and can also be mutated in place.
// TODO(adamg): This is a very bad name.
#[derive(Clone, Default, Debug)]
pub struct ArrayStats {
    inner: Arc<RwLock<StatsSet>>,
}

/// Reference to an array's [`StatsSet`]. Can be used to get and mutate the underlying stats.
///
/// Constructed by calling [`ArrayStats::to_ref`].
pub struct StatsSetRef<'a> {
    // We need to reference back to the array
    dyn_array_ref: &'a ArrayRef,
    array_stats: &'a ArrayStats,
}

impl ArrayStats {
    pub fn to_ref<'a>(&'a self, array: &'a ArrayRef) -> StatsSetRef<'a> {
        StatsSetRef {
            dyn_array_ref: array,
            array_stats: self,
        }
    }

    pub fn set(&self, stat: Stat, value: Precision<ScalarValue>) {
        self.inner.write().set(stat, value);
    }

    pub fn clear(&self, stat: Stat) {
        self.inner.write().clear(stat);
    }

    pub fn retain(&self, stats: &[Stat]) {
        self.inner.write().retain_only(stats);
    }
}

impl From<StatsSet> for ArrayStats {
    fn from(value: StatsSet) -> Self {
        Self {
            inner: Arc::new(RwLock::new(value)),
        }
    }
}

impl From<ArrayStats> for StatsSet {
    fn from(value: ArrayStats) -> Self {
        value.inner.read().clone()
    }
}

impl StatsSetRef<'_> {
    pub(crate) fn replace(&self, stats: StatsSet) {
        *self.array_stats.inner.write() = stats;
    }

    pub fn set_iter(&self, iter: StatsSetIntoIter) {
        let mut guard = self.array_stats.inner.write();
        for (stat, value) in iter {
            guard.set(stat, value);
        }
    }

    /// Copy every statistic of `stats` into this set, overwriting existing entries, without
    /// cloning the source set. A no-op when both refer to the same stats or the source is empty.
    pub(crate) fn transfer_from(&self, stats: &ArrayStats) {
        if Arc::ptr_eq(&self.array_stats.inner, &stats.inner) {
            return;
        }
        let source = stats.inner.read();
        if source.is_empty() {
            return;
        }
        let mut guard = self.array_stats.inner.write();
        for (stat, value) in source.iter() {
            guard.set(*stat, value.clone());
        }
    }

    pub fn inherit_from(&self, stats: StatsSetRef<'_>) {
        // Only inherit if the underlying stats are different
        if !Arc::ptr_eq(&self.array_stats.inner, &stats.array_stats.inner) {
            stats.with_iter(|iter| self.inherit(iter));
        }
    }

    pub fn inherit<'a>(&self, iter: impl Iterator<Item = &'a (Stat, Precision<ScalarValue>)>) {
        let mut guard = self.array_stats.inner.write();
        for (stat, value) in iter {
            if !value.is_exact() {
                if !guard.get(*stat).is_exact() {
                    guard.set(*stat, value.clone());
                }
            } else {
                guard.set(*stat, value.clone());
            }
        }
    }

    pub fn with_typed_stats_set<U, F: FnOnce(TypedStatsSetRef) -> U>(&self, apply: F) -> U {
        apply(
            self.array_stats
                .inner
                .read()
                .as_typed_ref(self.dyn_array_ref.dtype()),
        )
    }

    pub fn with_mut_typed_stats_set<U, F: FnOnce(MutTypedStatsSetRef) -> U>(&self, apply: F) -> U {
        apply(
            self.array_stats
                .inner
                .write()
                .as_mut_typed_ref(self.dyn_array_ref.dtype()),
        )
    }

    pub fn to_owned(&self) -> StatsSet {
        self.array_stats.inner.read().clone()
    }

    /// Returns a clone of the underlying [`ArrayStats`].
    ///
    /// Since [`ArrayStats`] uses `Arc` internally, this is a cheap reference-count increment.
    pub fn to_array_stats(&self) -> ArrayStats {
        self.array_stats.clone()
    }

    pub fn with_iter<
        F: for<'a> FnOnce(&mut dyn Iterator<Item = &'a (Stat, Precision<ScalarValue>)>) -> R,
        R,
    >(
        &self,
        f: F,
    ) -> R {
        let lock = self.array_stats.inner.read();
        f(&mut lock.iter())
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

    pub fn set(&self, stat: Stat, value: Precision<ScalarValue>) {
        self.array_stats.set(stat, value);
    }

    pub fn clear(&self, stat: Stat) {
        self.array_stats.clear(stat);
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
        self.array_stats
            .inner
            .read()
            .as_typed_ref(self.dyn_array_ref.dtype())
            .get(stat)
    }

    fn len(&self) -> usize {
        self.array_stats.inner.read().len()
    }
}

#[cfg(test)]
mod tests {
    use vortex_buffer::buffer;

    use crate::IntoArray;
    use crate::expr::stats::Precision;
    use crate::stats::Stat;

    #[test]
    fn transfer_from_copies_entries_without_touching_the_source() {
        let source = buffer![1i32, 2, 3].into_array();
        let target = buffer![4i32, 5, 6].into_array();
        source.statistics().set(Stat::Max, Precision::exact(3i32));
        target.statistics().set(Stat::Max, Precision::inexact(9i32));
        target.statistics().set(Stat::Min, Precision::exact(4i32));

        target
            .statistics()
            .transfer_from(&source.statistics().to_array_stats());

        assert_eq!(
            target.statistics().to_owned().get(Stat::Max),
            Precision::exact(3i32)
        );
        assert_eq!(
            target.statistics().to_owned().get(Stat::Min),
            Precision::exact(4i32)
        );
        assert_eq!(source.statistics().to_owned().len(), 1);
    }

    #[test]
    fn transfer_from_empty_or_self_is_a_no_op() {
        let empty = buffer![1i32].into_array();
        let target = buffer![2i32].into_array();
        target.statistics().set(Stat::Max, Precision::exact(2i32));

        target
            .statistics()
            .transfer_from(&empty.statistics().to_array_stats());
        target
            .statistics()
            .transfer_from(&target.statistics().to_array_stats());

        assert_eq!(target.statistics().to_owned().len(), 1);
        assert_eq!(
            target.statistics().to_owned().get(Stat::Max),
            Precision::exact(2i32)
        );
    }
}
