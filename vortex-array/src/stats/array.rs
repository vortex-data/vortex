// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stats as they are stored on arrays.

use std::sync::Arc;

use arc_swap::ArcSwap;
use vortex_array::ExecutionCtx;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;

use super::MutTypedStatsSetRef;
use super::StatsSet;
use super::StatsSetIntoIter;
use super::TypedStatsSetRef;
use crate::ArrayRef;
use crate::aggregate_fn::fns::is_constant::is_constant;
use crate::aggregate_fn::fns::is_sorted::is_sorted;
use crate::aggregate_fn::fns::is_sorted::is_strict_sorted;
use crate::aggregate_fn::fns::min_max::MinMaxResult;
use crate::aggregate_fn::fns::min_max::min_max;
use crate::aggregate_fn::fns::nan_count::nan_count;
use crate::aggregate_fn::fns::sum::sum;
use crate::builders::builder_with_capacity;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;
use crate::scalar::Scalar;
use crate::scalar::ScalarValue;

/// A shared [`StatsSet`] stored in an array. Can be shared by copies of the array and can also be mutated in place.
// TODO(adamg): This is a very bad name.
#[derive(Clone, Debug)]
pub struct ArrayStats {
    // Lock-free reads via copy-on-write. Writes are last-writer-wins;
    // concurrent writers may lose updates, which is acceptable for stats
    // (they're hints and can be recomputed).
    inner: Arc<ArcSwap<StatsSet>>,
}

impl Default for ArrayStats {
    fn default() -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(StatsSet::default())),
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

    pub fn set(&self, stat: Stat, value: Precision<ScalarValue>) {
        let mut new_stats = (**self.inner.load()).clone();
        new_stats.set(stat, value);
        self.inner.store(Arc::new(new_stats));
    }

    pub fn clear(&self, stat: Stat) {
        let mut new_stats = (**self.inner.load()).clone();
        new_stats.clear(stat);
        self.inner.store(Arc::new(new_stats));
    }

    pub fn retain(&self, stats: &[Stat]) {
        let mut new_stats = (**self.inner.load()).clone();
        new_stats.retain_only(stats);
        self.inner.store(Arc::new(new_stats));
    }
}

impl From<StatsSet> for ArrayStats {
    fn from(value: StatsSet) -> Self {
        Self {
            inner: Arc::new(ArcSwap::from_pointee(value)),
        }
    }
}

impl From<ArrayStats> for StatsSet {
    fn from(value: ArrayStats) -> Self {
        (**value.inner.load()).clone()
    }
}

impl StatsSetRef<'_> {
    pub(crate) fn replace(&self, stats: StatsSet) {
        self.array_stats.inner.store(Arc::new(stats));
    }

    pub fn set_iter(&self, iter: StatsSetIntoIter) {
        let mut new_stats = (**self.array_stats.inner.load()).clone();
        for (stat, value) in iter {
            new_stats.set(stat, value);
        }
        self.array_stats.inner.store(Arc::new(new_stats));
    }

    pub fn inherit_from(&self, stats: StatsSetRef<'_>) {
        // Only inherit if the underlying stats are different
        if !Arc::ptr_eq(&self.array_stats.inner, &stats.array_stats.inner) {
            stats.with_iter(|iter| self.inherit(iter));
        }
    }

    pub fn inherit<'a>(&self, iter: impl Iterator<Item = &'a (Stat, Precision<ScalarValue>)>) {
        let mut new_stats = (**self.array_stats.inner.load()).clone();
        for (stat, value) in iter {
            if !value.is_exact() {
                if !new_stats.get(*stat).is_some_and(|v| v.is_exact()) {
                    new_stats.set(*stat, value.clone());
                }
            } else {
                new_stats.set(*stat, value.clone());
            }
        }
        self.array_stats.inner.store(Arc::new(new_stats));
    }

    pub fn with_typed_stats_set<U, F: FnOnce(TypedStatsSetRef) -> U>(&self, apply: F) -> U {
        let snapshot = self.array_stats.inner.load();
        apply(snapshot.as_typed_ref(self.dyn_array_ref.dtype()))
    }

    pub fn with_mut_typed_stats_set<U, F: FnOnce(MutTypedStatsSetRef) -> U>(&self, apply: F) -> U {
        let mut new_stats = (**self.array_stats.inner.load()).clone();
        let result = apply(new_stats.as_mut_typed_ref(self.dyn_array_ref.dtype()));
        self.array_stats.inner.store(Arc::new(new_stats));
        result
    }

    pub fn to_owned(&self) -> StatsSet {
        (**self.array_stats.inner.load()).clone()
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
        let snapshot = self.array_stats.inner.load();
        f(&mut snapshot.iter())
    }

    pub fn compute_stat(&self, stat: Stat, ctx: &mut ExecutionCtx) -> VortexResult<Option<Scalar>> {
        // If it's already computed and exact, we can return it.
        if let Some(Precision::Exact(s)) = self.get(stat) {
            return Ok(Some(s));
        }

        Ok(match stat {
            Stat::Min => min_max(self.dyn_array_ref, ctx)?.map(|MinMaxResult { min, max: _ }| min),
            Stat::Max => min_max(self.dyn_array_ref, ctx)?.map(|MinMaxResult { min: _, max }| max),
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
            Stat::UncompressedSizeInBytes => {
                let mut builder =
                    builder_with_capacity(self.dyn_array_ref.dtype(), self.dyn_array_ref.len());
                unsafe {
                    builder.extend_from_array_unchecked(self.dyn_array_ref);
                }
                let nbytes = builder.finish().nbytes();
                self.set(stat, Precision::exact(nbytes));
                Some(nbytes.into())
            }
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
    fn get(&self, stat: Stat) -> Option<Precision<Scalar>> {
        self.array_stats
            .inner
            .load()
            .as_typed_ref(self.dyn_array_ref.dtype())
            .get(stat)
    }

    fn len(&self) -> usize {
        self.array_stats.inner.load().len()
    }
}
