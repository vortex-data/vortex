// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Stats as they are stored on arrays.

use std::sync::Arc;
use std::sync::OnceLock;

use parking_lot::RwLock;
use vortex_error::VortexError;
use vortex_error::VortexResult;
use vortex_error::vortex_panic;
use vortex_scalar::Scalar;
use vortex_scalar::ScalarValue;

use super::MutTypedStatsSetRef;
use super::StatsSet;
use super::StatsSetIntoIter;
use super::TypedStatsSetRef;
use crate::Array;
use crate::builders::builder_with_capacity;
use crate::compute::MinMaxResult;
use crate::compute::is_constant;
use crate::compute::is_sorted;
use crate::compute::is_strict_sorted;
use crate::compute::min_max;
use crate::compute::nan_count;
use crate::compute::sum;
use crate::expr::stats::Precision;
use crate::expr::stats::Stat;
use crate::expr::stats::StatsProvider;

/// A shared [`StatsSet`] stored in an array. Can be shared by copies of the array and can also be mutated in place.
///
/// Uses lazy allocation: the backing `Arc<RwLock<StatsSet>>` is only allocated on the first write.
/// This avoids heap allocation and deallocation overhead for ephemeral wrapper arrays
/// (Slice, Filter, ScalarFn) that never use stats.
// TODO(adamg): This is a very bad name.
#[derive(Clone, Default, Debug)]
pub struct ArrayStats {
    inner: OnceLock<Arc<RwLock<StatsSet>>>,
}

/// Reference to an array's [`StatsSet`]. Can be used to get and mutate the underlying stats.
///
/// Constructed by calling [`ArrayStats::to_ref`].
pub struct StatsSetRef<'a> {
    // We need to reference back to the array
    dyn_array_ref: &'a dyn Array,
    array_stats: &'a ArrayStats,
}

impl ArrayStats {
    /// Returns the inner Arc, allocating it on first access (for writes).
    fn get_or_init(&self) -> &Arc<RwLock<StatsSet>> {
        self.inner
            .get_or_init(|| Arc::new(RwLock::new(StatsSet::default())))
    }

    pub fn to_ref<'a>(&'a self, array: &'a dyn Array) -> StatsSetRef<'a> {
        StatsSetRef {
            dyn_array_ref: array,
            array_stats: self,
        }
    }

    pub fn set(&self, stat: Stat, value: Precision<ScalarValue>) {
        self.get_or_init().write().set(stat, value);
    }

    pub fn clear(&self, stat: Stat) {
        self.get_or_init().write().clear(stat);
    }

    pub fn retain(&self, stats: &[Stat]) {
        self.get_or_init().write().retain_only(stats);
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
        match value.inner.get() {
            Some(arc) => arc.read().clone(),
            None => StatsSet::default(),
        }
    }
}

impl StatsSetRef<'_> {
    pub fn set_iter(&self, iter: StatsSetIntoIter) {
        let mut guard = self.array_stats.get_or_init().write();
        for (stat, value) in iter {
            guard.set(stat, value);
        }
    }

    pub fn inherit_from(&self, stats: StatsSetRef<'_>) {
        match (self.array_stats.inner.get(), stats.array_stats.inner.get()) {
            // Same backing Arc, nothing to do.
            (Some(a), Some(b)) if Arc::ptr_eq(a, b) => {}
            // Source has no stats, nothing to inherit.
            (_, None) => {}
            // Otherwise, inherit from source.
            _ => stats.with_iter(|iter| self.inherit(iter)),
        }
    }

    pub fn inherit<'a>(&self, iter: impl Iterator<Item = &'a (Stat, Precision<ScalarValue>)>) {
        let mut guard = self.array_stats.get_or_init().write();
        for (stat, value) in iter {
            if !value.is_exact() {
                if !guard.get(*stat).is_some_and(|v| v.is_exact()) {
                    guard.set(*stat, value.clone());
                }
            } else {
                guard.set(*stat, value.clone());
            }
        }
    }

    pub fn with_typed_stats_set<U, F: FnOnce(TypedStatsSetRef) -> U>(&self, apply: F) -> U {
        match self.array_stats.inner.get() {
            Some(arc) => apply(arc.read().as_typed_ref(self.dyn_array_ref.dtype())),
            None => {
                let empty = StatsSet::default();
                apply(empty.as_typed_ref(self.dyn_array_ref.dtype()))
            }
        }
    }

    pub fn with_mut_typed_stats_set<U, F: FnOnce(MutTypedStatsSetRef) -> U>(&self, apply: F) -> U {
        apply(
            self.array_stats
                .get_or_init()
                .write()
                .as_mut_typed_ref(self.dyn_array_ref.dtype()),
        )
    }

    pub fn to_owned(&self) -> StatsSet {
        match self.array_stats.inner.get() {
            Some(arc) => arc.read().clone(),
            None => StatsSet::default(),
        }
    }

    pub fn with_iter<
        F: for<'a> FnOnce(&mut dyn Iterator<Item = &'a (Stat, Precision<ScalarValue>)>) -> R,
        R,
    >(
        &self,
        f: F,
    ) -> R {
        match self.array_stats.inner.get() {
            Some(arc) => {
                let lock = arc.read();
                f(&mut lock.iter())
            }
            None => f(&mut std::iter::empty()),
        }
    }

    pub fn compute_stat(&self, stat: Stat) -> VortexResult<Option<Scalar>> {
        // If it's already computed and exact, we can return it.
        if let Some(Precision::Exact(s)) = self.get(stat) {
            return Ok(Some(s));
        }

        Ok(match stat {
            Stat::Min => min_max(self.dyn_array_ref)?.map(|MinMaxResult { min, max: _ }| min),
            Stat::Max => min_max(self.dyn_array_ref)?.map(|MinMaxResult { min: _, max }| max),
            Stat::Sum => {
                Stat::Sum
                    .dtype(self.dyn_array_ref.dtype())
                    .is_some()
                    .then(|| {
                        // Sum is supported for this dtype.
                        sum(self.dyn_array_ref)
                    })
                    .transpose()?
            }
            Stat::NullCount => self.dyn_array_ref.invalid_count().ok().map(Into::into),
            Stat::IsConstant => {
                if self.dyn_array_ref.is_empty() {
                    None
                } else {
                    is_constant(self.dyn_array_ref)?.map(|v| v.into())
                }
            }
            Stat::IsSorted => is_sorted(self.dyn_array_ref)?.map(|v| v.into()),
            Stat::IsStrictSorted => is_strict_sorted(self.dyn_array_ref)?.map(|v| v.into()),
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
                        nan_count(self.dyn_array_ref)
                    })
                    .transpose()?
                    .map(|s| s.into())
            }
        })
    }

    pub fn compute_all(&self, stats: &[Stat]) -> VortexResult<StatsSet> {
        let mut stats_set = StatsSet::default();
        for &stat in stats {
            if let Some(s) = self.compute_stat(stat)?
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
    ) -> Option<U> {
        self.compute_stat(stat)
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

    pub fn compute_min<U: for<'a> TryFrom<&'a Scalar, Error = VortexError>>(&self) -> Option<U> {
        self.compute_as(Stat::Min)
    }

    pub fn compute_max<U: for<'a> TryFrom<&'a Scalar, Error = VortexError>>(&self) -> Option<U> {
        self.compute_as(Stat::Max)
    }

    pub fn compute_is_sorted(&self) -> Option<bool> {
        self.compute_as(Stat::IsSorted)
    }

    pub fn compute_is_strict_sorted(&self) -> Option<bool> {
        self.compute_as(Stat::IsStrictSorted)
    }

    pub fn compute_is_constant(&self) -> Option<bool> {
        self.compute_as(Stat::IsConstant)
    }

    pub fn compute_null_count(&self) -> Option<usize> {
        self.compute_as(Stat::NullCount)
    }

    pub fn compute_uncompressed_size_in_bytes(&self) -> Option<usize> {
        self.compute_as(Stat::UncompressedSizeInBytes)
    }
}

impl StatsProvider for StatsSetRef<'_> {
    fn get(&self, stat: Stat) -> Option<Precision<Scalar>> {
        self.array_stats
            .inner
            .get()?
            .read()
            .as_typed_ref(self.dyn_array_ref.dtype())
            .get(stat)
    }

    fn len(&self) -> usize {
        match self.array_stats.inner.get() {
            Some(arc) => arc.read().len(),
            None => 0,
        }
    }
}
