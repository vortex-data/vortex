// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::atomic::AtomicUsize;
use std::sync::atomic::Ordering;

use vortex_expr::BoundExpression;
use vortex_expr::ExactKey;
use vortex_expr::WeakBoundExprKey;
use vortex_utils::aliases::dash_map::DashMap;

/// Concurrent cache keyed by [`BoundExpression`] identity whose entries do not keep expressions
/// alive.
///
/// Hits probe by borrowed expression identity under a shared shard lock. Every
/// [`Self::EVICT_INTERVAL`] misses, one miss runs [`DashMap::retain`] to drop entries whose
/// expressions no scan holds anymore, bounding dead keys on long-lived layout readers without
/// write-locking every shard on every miss.
#[derive(Debug)]
pub struct WeakExprCache<V> {
    map: DashMap<WeakBoundExprKey, V>,
    misses: AtomicUsize,
}

impl<V> Default for WeakExprCache<V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<V> WeakExprCache<V> {
    /// Number of cache misses between dead-key eviction sweeps.
    pub const EVICT_INTERVAL: usize = 64;

    /// Creates an empty cache with the default shard count.
    pub fn new() -> Self {
        Self {
            map: DashMap::default(),
            misses: AtomicUsize::new(0),
        }
    }

    /// Creates an empty cache with the given shard count.
    pub fn with_shard_amount(shard_amount: usize) -> Self {
        Self {
            map: DashMap::with_hasher_and_shard_amount(Default::default(), shard_amount),
            misses: AtomicUsize::new(0),
        }
    }

    /// Returns a cloned value for `expr` if present, without cloning `expr`.
    pub fn get_cloned(&self, expr: &BoundExpression) -> Option<V>
    where
        V: Clone,
    {
        self.map
            .get(expr as &dyn ExactKey)
            .map(|entry| entry.value().clone())
    }

    /// Returns the cached value for `expr`, inserting `init()` on a miss.
    pub fn get_or_insert_with(&self, expr: &BoundExpression, init: impl FnOnce() -> V) -> V
    where
        V: Clone,
    {
        if let Some(entry) = self.map.get(expr as &dyn ExactKey) {
            return entry.value().clone();
        }

        self.maybe_evict();
        self.map
            .entry(WeakBoundExprKey::new(expr))
            .or_insert_with(init)
            .value()
            .clone()
    }

    /// Runs `f` on the cached entry for `expr`, inserting `init()` on a miss.
    pub fn with_entry_mut<R>(
        &self,
        expr: &BoundExpression,
        init: impl FnOnce() -> V,
        f: impl FnOnce(&mut V) -> R,
    ) -> R {
        if let Some(mut entry) = self.map.get_mut(expr as &dyn ExactKey) {
            return f(&mut entry);
        }

        self.maybe_evict();
        let mut entry = self
            .map
            .entry(WeakBoundExprKey::new(expr))
            .or_insert_with(init);
        f(&mut entry)
    }

    /// Returns the number of entries currently stored, including not-yet-evicted dead keys.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// Returns `true` if the cache contains no entries.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    fn maybe_evict(&self) {
        if self.misses.fetch_add(1, Ordering::Relaxed) % Self::EVICT_INTERVAL
            == Self::EVICT_INTERVAL - 1
        {
            self.map.retain(|key, _| key.is_live());
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_array::dtype::PType;
    use vortex_error::VortexResult;
    use vortex_expr::gt;
    use vortex_expr::lit;
    use vortex_expr::root;

    use super::WeakExprCache;

    #[test]
    fn evicts_dead_entries_periodically_without_dropping_live_ones() -> VortexResult<()> {
        let dtype = DType::Primitive(PType::I32, Nullability::NonNullable);
        let cache = WeakExprCache::<usize>::new();

        let live = gt(root(), lit(0i32)).bind(&dtype)?;
        assert_eq!(cache.get_or_insert_with(&live, || 1), 1);

        // Hits do not count as misses or trigger eviction.
        for _ in 0..(WeakExprCache::<usize>::EVICT_INTERVAL * 2) {
            assert_eq!(cache.get_cloned(&live), Some(1));
        }

        // Insert dead expressions up to one short of the sweep interval: none evicted yet.
        for i in 1..(WeakExprCache::<usize>::EVICT_INTERVAL - 1) {
            let dead = gt(root(), lit(i32::try_from(i)?)).bind(&dtype)?;
            assert_eq!(cache.get_or_insert_with(&dead, || i + 1), i + 1);
        }
        assert_eq!(cache.len(), WeakExprCache::<usize>::EVICT_INTERVAL - 1);

        // The EVICT_INTERVAL-th miss sweeps all dead keys while keeping the live entry.
        let trigger = gt(root(), lit(-1i32)).bind(&dtype)?;
        assert_eq!(cache.get_or_insert_with(&trigger, || 999), 999);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.get_cloned(&live), Some(1));
        assert_eq!(cache.get_cloned(&trigger), Some(999));

        Ok(())
    }
}
