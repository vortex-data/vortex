// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Scoped, bounded cache for canonicalized arrays used by the `execute_scalar` fallback path.
//!
//! When an encoding has no registered scalar-extraction kernel, the dispatcher canonicalizes the
//! array once and reuses the canonical form for subsequent scalar reads. The cache is owned by
//! the [`ExecutionCtx`](crate::ExecutionCtx); see [`ExecutionCtx::fork_shared`] and
//! [`ExecutionCtx::fork_isolated`] for how cache identity propagates when a context is forked.
//!
//! Keys are the `Arc` identity of the source [`ArrayRef`] (see [`ArrayRef::addr`]). Two slices
//! of the same underlying array have distinct keys; encodings that wrap a child (`Slice`,
//! `Filter`, ...) can choose to forward to the child's cache slot for better reuse.

use std::collections::VecDeque;
use std::num::NonZeroUsize;
use std::sync::Arc;

use parking_lot::Mutex;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::Canonical;

/// Identity key for a cached entry: the address of the source `ArrayRef`'s `Arc`.
pub type CacheKey = usize;

/// Bound configuration for [`ScalarAtCache`].
#[derive(Debug, Clone, Copy)]
pub struct ScalarAtCacheConfig {
    /// Maximum number of cached entries before FIFO eviction begins.
    pub max_entries: NonZeroUsize,
}

impl ScalarAtCacheConfig {
    /// Default bound: 64 cached canonical arrays per context.
    pub const DEFAULT: Self = Self {
        max_entries: match NonZeroUsize::new(64) {
            Some(n) => n,
            None => unreachable!(),
        },
    };
}

impl Default for ScalarAtCacheConfig {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[derive(Debug)]
struct CachedEntry {
    canonical: Canonical,
    pin_count: usize,
}

#[derive(Debug)]
struct Inner {
    entries: HashMap<CacheKey, CachedEntry>,
    order: VecDeque<CacheKey>,
    config: ScalarAtCacheConfig,
}

impl Inner {
    fn new(config: ScalarAtCacheConfig) -> Self {
        Self {
            entries: HashMap::default(),
            order: VecDeque::new(),
            config,
        }
    }

    /// Evict unpinned entries from the front of the FIFO until the entry count is within bound.
    /// Pinned entries are skipped; if all remaining entries are pinned, eviction stops.
    fn evict_to_capacity(&mut self) {
        let cap = self.config.max_entries.get();
        if self.entries.len() <= cap {
            return;
        }
        let mut scanned = 0usize;
        let max_scan = self.order.len();
        while self.entries.len() > cap && scanned < max_scan {
            let Some(candidate) = self.order.pop_front() else {
                break;
            };
            scanned += 1;
            match self.entries.get(&candidate) {
                Some(e) if e.pin_count == 0 => {
                    self.entries.remove(&candidate);
                }
                Some(_) => {
                    // Pinned — keep in cache, re-queue at back so we can revisit later.
                    self.order.push_back(candidate);
                }
                None => {
                    // Stale order entry; skip without re-queueing.
                }
            }
        }
    }
}

/// A scoped, bounded cache of canonicalized arrays.
///
/// The cache is `Clone`able; clones share the same underlying storage via `Arc`. Use
/// [`ScalarAtCache::fork_isolated`] to construct a sibling cache with independent storage and
/// the same configuration.
#[derive(Clone, Debug)]
pub struct ScalarAtCache {
    inner: Arc<Mutex<Inner>>,
}

impl ScalarAtCache {
    /// Construct a new empty cache with the given configuration.
    pub fn new(config: ScalarAtCacheConfig) -> Self {
        Self {
            inner: Arc::new(Mutex::new(Inner::new(config))),
        }
    }

    /// Construct a sibling cache with the same configuration but independent storage.
    pub fn fork_isolated(&self) -> Self {
        let config = self.inner.lock().config;
        Self::new(config)
    }

    /// Returns the configuration this cache was constructed with.
    pub fn config(&self) -> ScalarAtCacheConfig {
        self.inner.lock().config
    }

    /// Returns the current number of cached entries.
    pub fn len(&self) -> usize {
        self.inner.lock().entries.len()
    }

    /// Returns whether the cache currently holds any entries.
    pub fn is_empty(&self) -> bool {
        self.inner.lock().entries.is_empty()
    }

    /// Drop all cached entries. Pin counts are also cleared.
    pub fn clear(&self) {
        let mut inner = self.inner.lock();
        inner.entries.clear();
        inner.order.clear();
    }

    /// Look up `key`; on hit return a clone of the cached canonical. On miss invoke `f` to
    /// canonicalize, insert the result, and return it.
    ///
    /// `f` is invoked with the cache lock released so it may itself perform compute that touches
    /// the same cache without deadlocking. Concurrent misses on the same key may race and both
    /// run `f`; the first insertion wins and the second is discarded, which is safe because
    /// canonical form is value-equal regardless of which call produced it.
    pub fn get_or_canonicalize<F>(&self, key: CacheKey, f: F) -> VortexResult<Canonical>
    where
        F: FnOnce() -> VortexResult<Canonical>,
    {
        if let Some(hit) = self.try_get(key) {
            return Ok(hit);
        }
        let canonical = f()?;
        self.insert(key, canonical.clone());
        Ok(canonical)
    }

    /// Look up `key`, returning a clone of the cached canonical on hit.
    pub fn try_get(&self, key: CacheKey) -> Option<Canonical> {
        let inner = self.inner.lock();
        inner.entries.get(&key).map(|e| e.canonical.clone())
    }

    /// Insert `canonical` at `key`. If an entry already exists it is overwritten. Eviction runs
    /// after insertion to bring the cache back within capacity, skipping pinned entries.
    pub fn insert(&self, key: CacheKey, canonical: Canonical) {
        let mut inner = self.inner.lock();
        let already_present = inner.entries.contains_key(&key);
        if let Some(existing) = inner.entries.get_mut(&key) {
            existing.canonical = canonical;
        } else {
            inner.entries.insert(
                key,
                CachedEntry {
                    canonical,
                    pin_count: 0,
                },
            );
        }
        if !already_present {
            inner.order.push_back(key);
            inner.evict_to_capacity();
        }
    }

    /// Increment the pin count for `key`. Pinned entries are not evicted by capacity pressure.
    /// Pinning a key that is not yet present is a no-op; the pin must be re-applied after the
    /// entry is inserted.
    pub fn pin(&self, key: CacheKey) {
        let mut inner = self.inner.lock();
        if let Some(entry) = inner.entries.get_mut(&key) {
            entry.pin_count = entry.pin_count.saturating_add(1);
        }
    }

    /// Decrement the pin count for `key`. Unpinning below zero saturates at zero.
    pub fn unpin(&self, key: CacheKey) {
        let mut inner = self.inner.lock();
        if let Some(entry) = inner.entries.get_mut(&key) {
            entry.pin_count = entry.pin_count.saturating_sub(1);
        }
    }
}

impl Default for ScalarAtCache {
    fn default() -> Self {
        Self::new(ScalarAtCacheConfig::DEFAULT)
    }
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroUsize;

    use vortex_error::VortexResult;

    use super::ScalarAtCache;
    use super::ScalarAtCacheConfig;
    use crate::Canonical;
    use crate::arrays::primitive::PrimitiveArray;
    use crate::validity::Validity;

    fn canonical_primitive(values: &[i32]) -> Canonical {
        Canonical::Primitive(PrimitiveArray::new(values.to_vec(), Validity::NonNullable))
    }

    #[test]
    fn miss_then_hit() -> VortexResult<()> {
        let cache = ScalarAtCache::default();
        let computed = std::cell::Cell::new(0u32);
        let key: usize = 0xDEAD;

        let c1 = cache.get_or_canonicalize(key, || {
            computed.set(computed.get() + 1);
            Ok(canonical_primitive(&[1, 2, 3]))
        })?;
        let c2 = cache.get_or_canonicalize(key, || {
            computed.set(computed.get() + 1);
            Ok(canonical_primitive(&[9, 9, 9]))
        })?;

        assert_eq!(computed.get(), 1, "miss should canonicalize exactly once");
        drop((c1, c2));
        assert_eq!(cache.len(), 1);
        Ok(())
    }

    #[test]
    fn fifo_evicts_unpinned() -> VortexResult<()> {
        let config = ScalarAtCacheConfig {
            max_entries: NonZeroUsize::new(2).unwrap(),
        };
        let cache = ScalarAtCache::new(config);
        cache.insert(1, canonical_primitive(&[1]));
        cache.insert(2, canonical_primitive(&[2]));
        cache.insert(3, canonical_primitive(&[3]));
        assert_eq!(cache.len(), 2);
        assert!(cache.try_get(1).is_none());
        assert!(cache.try_get(2).is_some());
        assert!(cache.try_get(3).is_some());
        Ok(())
    }

    #[test]
    fn pin_blocks_eviction() -> VortexResult<()> {
        let config = ScalarAtCacheConfig {
            max_entries: NonZeroUsize::new(2).unwrap(),
        };
        let cache = ScalarAtCache::new(config);
        cache.insert(1, canonical_primitive(&[1]));
        cache.pin(1);
        cache.insert(2, canonical_primitive(&[2]));
        cache.insert(3, canonical_primitive(&[3]));
        assert!(cache.try_get(1).is_some(), "pinned entry must survive");
        cache.unpin(1);
        cache.insert(4, canonical_primitive(&[4]));
        // After unpin and another insert, key 1 is eligible for eviction.
        assert_eq!(cache.len(), 2);
        Ok(())
    }

    #[test]
    fn fork_isolated_is_independent() -> VortexResult<()> {
        let a = ScalarAtCache::default();
        a.insert(1, canonical_primitive(&[1]));
        let b = a.fork_isolated();
        assert!(b.try_get(1).is_none());
        b.insert(2, canonical_primitive(&[2]));
        assert!(a.try_get(2).is_none());
        Ok(())
    }

    #[test]
    fn clone_is_shared() -> VortexResult<()> {
        let a = ScalarAtCache::default();
        let b = a.clone();
        a.insert(1, canonical_primitive(&[1]));
        assert!(b.try_get(1).is_some(), "clone shares storage");
        b.clear();
        assert!(a.try_get(1).is_none());
        Ok(())
    }
}
