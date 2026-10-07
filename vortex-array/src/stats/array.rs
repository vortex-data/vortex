// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stats as they are stored on arrays.

use std::sync::Arc;
use std::sync::OnceLock;

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
/// Stats are created lazily on read or write.
// TODO(adamg): This is a very bad name.
#[derive(Default, Debug)]
pub struct ArrayStats {
    inner: OnceLock<Arc<RwLock<StatsSet>>>,
}

impl Clone for ArrayStats {
    fn clone(&self) -> Self {
        Self {
            inner: OnceLock::from(Arc::clone(self.shared())),
        }
    }
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

    fn shared(&self) -> &Arc<RwLock<StatsSet>> {
        self.inner.get_or_init(Default::default)
    }

    fn read<R>(&self, f: impl FnOnce(&StatsSet) -> R) -> R {
        match self.inner.get() {
            Some(shared) => f(&shared.read()),
            None => f(&StatsSet::default()),
        }
    }

    pub fn set(&self, stat: Stat, value: Precision<ScalarValue>) {
        self.shared().write().set(stat, value);
    }

    pub fn clear(&self, stat: Stat) {
        if let Some(shared) = self.inner.get() {
            shared.write().clear(stat);
        }
    }

    pub fn retain(&self, stats: &[Stat]) {
        if let Some(shared) = self.inner.get() {
            shared.write().retain_only(stats);
        }
    }
}

impl From<StatsSet> for ArrayStats {
    fn from(value: StatsSet) -> Self {
        Self {
            inner: OnceLock::from(Arc::new(RwLock::new(value))),
        }
    }
}

impl From<ArrayStats> for StatsSet {
    fn from(value: ArrayStats) -> Self {
        value.read(|stats| stats.clone())
    }
}

impl StatsSetRef<'_> {
    pub(crate) fn replace(&self, stats: StatsSet) {
        if stats.is_empty() && self.array_stats.inner.get().is_none() {
            return;
        }
        *self.array_stats.shared().write() = stats;
    }

    pub fn set_iter(&self, iter: StatsSetIntoIter) {
        let mut iter = iter.peekable();
        if iter.peek().is_none() {
            return;
        }
        let mut guard = self.array_stats.shared().write();
        for (stat, value) in iter {
            guard.set(stat, value);
        }
    }

    pub fn inherit_from(&self, stats: StatsSetRef<'_>) {
        let Some(source) = stats.array_stats.inner.get() else {
            return;
        };
        // Only inherit if the underlying stats are different
        if self
            .array_stats
            .inner
            .get()
            .is_none_or(|shared| !Arc::ptr_eq(shared, source))
        {
            stats.with_iter(|iter| self.inherit(iter));
        }
    }

    pub fn inherit<'a>(&self, iter: impl Iterator<Item = &'a (Stat, Precision<ScalarValue>)>) {
        let mut iter = iter.peekable();
        if iter.peek().is_none() {
            return;
        }
        let mut guard = self.array_stats.shared().write();
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
        self.array_stats
            .read(|stats| apply(stats.as_typed_ref(self.dyn_array_ref.dtype())))
    }

    pub fn with_mut_typed_stats_set<U, F: FnOnce(MutTypedStatsSetRef) -> U>(&self, apply: F) -> U {
        apply(
            self.array_stats
                .shared()
                .write()
                .as_mut_typed_ref(self.dyn_array_ref.dtype()),
        )
    }

    pub fn to_owned(&self) -> StatsSet {
        self.array_stats.read(|stats| stats.clone())
    }

    /// Returns a clone of the underlying [`ArrayStats`].
    pub fn to_array_stats(&self) -> ArrayStats {
        self.array_stats.clone()
    }

    /// Share underlying stats if they exist or return a lazy instance
    pub(crate) fn share_existing(&self) -> ArrayStats {
        ArrayStats {
            inner: match self.array_stats.inner.get() {
                Some(shared) => OnceLock::from(Arc::clone(shared)),
                None => OnceLock::new(),
            },
        }
    }

    pub fn with_iter<
        F: for<'a> FnOnce(&mut dyn Iterator<Item = &'a (Stat, Precision<ScalarValue>)>) -> R,
        R,
    >(
        &self,
        f: F,
    ) -> R {
        self.array_stats.read(|stats| f(&mut stats.iter()))
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
        self.compute_bool(Stat::IsSorted, ctx)
    }

    pub fn compute_is_strict_sorted(&self, ctx: &mut ExecutionCtx) -> Option<bool> {
        self.compute_bool(Stat::IsStrictSorted, ctx)
    }

    pub fn compute_is_constant(&self, ctx: &mut ExecutionCtx) -> Option<bool> {
        self.compute_bool(Stat::IsConstant, ctx)
    }

    pub fn compute_null_count(&self, ctx: &mut ExecutionCtx) -> Option<usize> {
        self.compute_usize(Stat::NullCount, ctx)
    }

    pub fn compute_uncompressed_size_in_bytes(&self, ctx: &mut ExecutionCtx) -> Option<usize> {
        self.compute_usize(Stat::UncompressedSizeInBytes, ctx)
    }

    /// Like [`Self::compute_as`], but reads a cached exact value without building a [`Scalar`].
    fn compute_bool(&self, stat: Stat, ctx: &mut ExecutionCtx) -> Option<bool> {
        // Bind the cached value first, so the read lock is released before any computation.
        let cached = self.array_stats.read(|stats| stats.get_bool(stat));
        if let Precision::Exact(value) = cached {
            return Some(value);
        }

        self.compute_as(stat, ctx)
    }

    /// Like [`Self::compute_as`], but reads a cached exact value without building a [`Scalar`].
    fn compute_usize(&self, stat: Stat, ctx: &mut ExecutionCtx) -> Option<usize> {
        // Bind the cached value first, so the read lock is released before any computation.
        let cached = self.array_stats.read(|stats| stats.get_usize(stat));
        if let Precision::Exact(value) = cached {
            return Some(value);
        }

        self.compute_as(stat, ctx)
    }
}

impl StatsProvider for StatsSetRef<'_> {
    fn get(&self, stat: Stat) -> Precision<Scalar> {
        self.array_stats
            .read(|stats| stats.as_typed_ref(self.dyn_array_ref.dtype()).get(stat))
    }

    fn len(&self) -> usize {
        self.array_stats.read(|stats| stats.len())
    }
}

#[cfg(test)]
mod tests {
    use super::ArrayStats;
    use super::StatsSet;
    use crate::expr::stats::Precision;
    use crate::expr::stats::Stat;
    use crate::scalar::ScalarValue;

    #[test]
    fn empty_stats() {
        let stats = ArrayStats::default();
        assert_eq!(StatsSet::from(stats).len(), 0);
    }

    #[test]
    fn clone_stats() {
        let stats = ArrayStats::default();
        let copy = stats.clone();

        stats.set(Stat::NullCount, Precision::exact(ScalarValue::from(3u64)));
        let through_copy = StatsSet::from(copy);
        assert_eq!(
            through_copy.get(Stat::NullCount),
            Precision::exact(ScalarValue::from(3u64))
        );
    }

    #[test]
    fn clone_writes() {
        let stats = ArrayStats::default();
        let copy = stats.clone();

        copy.set(Stat::NullCount, Precision::exact(ScalarValue::from(7u64)));
        assert_eq!(
            StatsSet::from(stats).get(Stat::NullCount),
            Precision::exact(ScalarValue::from(7u64))
        );
    }
}
