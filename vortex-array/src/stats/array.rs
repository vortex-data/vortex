//! Stats as they are stored on arrays.

use std::sync::Arc;

use parking_lot::RwLock;
use vortex_error::{VortexError, VortexResult, vortex_panic};
use vortex_scalar::ScalarValue;

use super::{
    Precision, Stat, StatType, StatsProvider, StatsProviderExt, StatsSet, StatsSetIntoIter,
};
use crate::Array;
use crate::compute::{MinMaxResult, is_constant, min_max, sum, uncompressed_size};

/// A shared [`StatsSet`] stored in an array. Can be shared by copies of the array and can also be mutated in place.
// TODO(adamg): This is a very bad name.
#[derive(Clone, Default, Debug)]
pub struct ArrayStats {
    inner: Arc<RwLock<StatsSet>>,
}

/// Reference to an array's [`StatsSet`]. Can be used to get and mutate the underlying stats.
pub struct StatsSetRef<'a> {
    // We need to reference back to the array
    dyn_array_ref: &'a dyn Array,
    parent_stats: ArrayStats,
}

impl ArrayStats {
    pub fn to_ref<'a>(&self, array: &'a dyn Array) -> StatsSetRef<'a> {
        StatsSetRef {
            dyn_array_ref: array,
            parent_stats: self.clone(),
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

impl StatsProvider for ArrayStats {
    fn get(&self, stat: Stat) -> Option<Precision<ScalarValue>> {
        let guard = self.inner.read();
        guard.get(stat)
    }

    fn len(&self) -> usize {
        let guard = self.inner.read();
        guard.len()
    }
}

impl StatsSetRef<'_> {
    pub fn set_iter(&self, iter: StatsSetIntoIter) {
        let mut guard = self.parent_stats.inner.write();

        for (stat, value) in iter {
            guard.set(stat, value);
        }
    }

    pub fn inherit(&self, parent_stats: StatsSetRef<'_>) {
        // TODO(ngates): depending on statistic, this should choose the more precise one
        self.set_iter(parent_stats.into_iter());
    }

    // TODO(adamg): potentially problematic name
    pub fn to_owned(&self) -> StatsSet {
        self.parent_stats.inner.read().clone()
    }

    pub fn into_iter(&self) -> StatsSetIntoIter {
        self.to_owned().into_iter()
    }

    pub fn compute_stat(&self, stat: Stat) -> VortexResult<Option<ScalarValue>> {
        // If it's already computed and exact, we can return it.
        if let Some(Precision::Exact(stat)) = self.get(stat) {
            return Ok(Some(stat));
        }

        // NOTE(ngates): this is the beginning of the stats refactor that pushes stats compute into
        //  regular compute functions.
        Ok(match stat {
            Stat::Min => {
                min_max(self.dyn_array_ref)?.map(|MinMaxResult { min, max: _ }| min.into_value())
            }
            Stat::Max => {
                min_max(self.dyn_array_ref)?.map(|MinMaxResult { min: _, max }| max.into_value())
            }
            Stat::Sum => {
                Stat::Sum
                    .dtype(self.dyn_array_ref.dtype())
                    .is_some()
                    .then(|| {
                        // Sum is supported for this dtype.
                        sum(self.dyn_array_ref)
                    })
                    .transpose()?
                    .map(|s| s.into_value())
            }
            Stat::NullCount => Some(self.dyn_array_ref.invalid_count()?.into()),
            Stat::IsConstant => {
                if self.dyn_array_ref.is_empty() {
                    None
                } else {
                    Some(is_constant(self.dyn_array_ref)?.into())
                }
            }
            Stat::UncompressedSizeInBytes => Some(uncompressed_size(self.dyn_array_ref)?.into()),
            _ => {
                let vtable = self.dyn_array_ref.vtable();
                let stats_set = vtable.compute_statistics(self.dyn_array_ref, stat)?;
                // Update the stats set with all the computed stats.
                for (stat, value) in stats_set.into_iter() {
                    self.set(stat, value);
                }
                self.get(stat).and_then(|p| p.as_exact())
            }
        })
    }

    pub fn compute_all(&self, stats: &[Stat]) -> VortexResult<StatsSet> {
        let mut stats_set = StatsSet::default();
        for stat in stats {
            if let Some(s) = self.compute_stat(*stat)? {
                stats_set.set(*stat, Precision::exact(s))
            }
        }
        Ok(stats_set)
    }
}

impl StatsSetRef<'_> {
    pub fn get_as<U: for<'a> TryFrom<&'a ScalarValue, Error = VortexError>>(
        &self,
        stat: Stat,
    ) -> Option<Precision<U>> {
        StatsProviderExt::get_as::<U>(self, stat)
    }

    pub fn get_as_bound<S, U>(&self) -> Option<S::Bound>
    where
        S: StatType<U>,
        U: for<'a> TryFrom<&'a ScalarValue, Error = VortexError>,
    {
        StatsProviderExt::get_as_bound::<S, U>(self)
    }

    pub fn compute_as<U: for<'a> TryFrom<&'a ScalarValue, Error = VortexError>>(
        &self,
        stat: Stat,
    ) -> Option<U> {
        self.compute_stat(stat)
            .inspect_err(|e| log::warn!("Failed to compute stat {}: {}", stat, e))
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
        self.parent_stats.set(stat, value);
    }

    pub fn clear(&self, stat: Stat) {
        self.parent_stats.clear(stat);
    }

    pub fn retain(&self, stats: &[Stat]) {
        self.parent_stats.retain(stats);
    }

    pub fn compute_min<U: for<'a> TryFrom<&'a ScalarValue, Error = VortexError>>(
        &self,
    ) -> Option<U> {
        self.compute_as(Stat::Min)
    }

    pub fn compute_max<U: for<'a> TryFrom<&'a ScalarValue, Error = VortexError>>(
        &self,
    ) -> Option<U> {
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
    fn get(&self, stat: Stat) -> Option<Precision<ScalarValue>> {
        self.parent_stats.get(stat)
    }

    fn len(&self) -> usize {
        self.parent_stats.len()
    }
}
