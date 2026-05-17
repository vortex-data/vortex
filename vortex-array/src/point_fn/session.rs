// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`PointSession`] — the caching point-fn dispatcher.

use std::collections::VecDeque;

use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::point_fn::BlockKey;
use crate::point_fn::PointDispatch;
use crate::point_fn::dispatch::AnyBlock;
use crate::point_fn::dispatch::CacheArrayId;
use crate::scalar::Scalar;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSortedSide;

/// Default capacity for the per-session scalar cache.
pub(crate) const DEFAULT_SCALAR_CACHE_CAPACITY: usize = 64;

/// Default capacity for the per-session decoded-block cache.
pub(crate) const DEFAULT_BLOCK_CACHE_CAPACITY: usize = 8;

/// A caching point-fn dispatcher. Holds a bounded scalar cache and a bounded
/// decoded-block cache that persist for the lifetime of the session.
///
/// Internal implementation detail; users go through
/// [`RepeatedAccess`](super::RepeatedAccess) which wraps a session.
///
/// ## Eviction policy
///
/// Phase 1 uses FIFO eviction with a fixed capacity. A future change may upgrade
/// to LRU; the public API does not assume a specific policy.
pub(crate) struct PointSession<'a> {
    ctx: &'a mut ExecutionCtx,
    scalar_cache: BoundedFifo<(CacheArrayId, usize), Scalar>,
    block_cache: BoundedFifo<(CacheArrayId, BlockKey), AnyBlock>,
}

impl<'a> PointSession<'a> {
    pub(crate) fn new(ctx: &'a mut ExecutionCtx) -> Self {
        Self::with_capacities(
            ctx,
            DEFAULT_SCALAR_CACHE_CAPACITY,
            DEFAULT_BLOCK_CACHE_CAPACITY,
        )
    }

    pub(crate) fn with_capacities(
        ctx: &'a mut ExecutionCtx,
        scalar_capacity: usize,
        block_capacity: usize,
    ) -> Self {
        Self {
            ctx,
            scalar_cache: BoundedFifo::new(scalar_capacity),
            block_cache: BoundedFifo::new(block_capacity),
        }
    }

    pub(crate) fn scalar_cache_len(&self) -> usize {
        self.scalar_cache.len()
    }

    pub(crate) fn block_cache_len(&self) -> usize {
        self.block_cache.len()
    }
}

impl PointDispatch for PointSession<'_> {
    fn ctx(&mut self) -> &mut ExecutionCtx {
        self.ctx
    }

    fn scalar_at(&mut self, arr: &ArrayRef, idx: usize) -> VortexResult<Scalar> {
        let key = (arr.addr(), idx);
        if let Some(v) = self.scalar_cache.get(&key) {
            return Ok(v.clone());
        }
        // Route through the encoding's `point_scalar_at` vtable hook (default:
        // forward to scalar_at). View encodings recurse via d.scalar_at which
        // re-enters this method at the child level, so the cache applies at
        // every level of the tree.
        let v = arr.point_execute_scalar(idx, self)?;
        self.scalar_cache.put(key, v.clone());
        Ok(v)
    }

    fn search_sorted(
        &mut self,
        arr: &ArrayRef,
        value: &Scalar,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        arr.point_execute_search_sorted(value, side, self)
    }

    fn cached_block_dyn(
        &mut self,
        key: (CacheArrayId, BlockKey),
        decode: &mut dyn FnMut() -> VortexResult<AnyBlock>,
    ) -> VortexResult<AnyBlock> {
        if let Some(any) = self.block_cache.get(&key) {
            return Ok(std::sync::Arc::<dyn std::any::Any + Send + Sync>::clone(
                any,
            ));
        }
        let v = decode()?;
        self.block_cache.put(
            key,
            std::sync::Arc::<dyn std::any::Any + Send + Sync>::clone(&v),
        );
        Ok(v)
    }
}

/// Bounded FIFO map: insert evicts the oldest entry when at capacity.
/// Simpler than LRU and adequate for Phase 1; can be upgraded later.
struct BoundedFifo<K, V> {
    map: HashMap<K, V>,
    insertion_order: VecDeque<K>,
    capacity: usize,
}

impl<K: std::hash::Hash + Eq + Clone, V> BoundedFifo<K, V> {
    fn new(capacity: usize) -> Self {
        Self {
            map: HashMap::with_capacity(capacity),
            insertion_order: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    fn len(&self) -> usize {
        self.map.len()
    }

    fn get(&self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    // `contains_key` then `insert` is intentional: we need to check membership
    // before deciding whether to evict, which doesn't fit clippy's `Entry`
    // suggestion (the eviction path would invalidate the entry borrow).
    #[allow(clippy::map_entry)]
    fn put(&mut self, key: K, value: V) {
        if self.capacity == 0 {
            return;
        }
        let was_present = self.map.contains_key(&key);
        if !was_present {
            if self.insertion_order.len() >= self.capacity
                && let Some(oldest) = self.insertion_order.pop_front()
            {
                self.map.remove(&oldest);
            }
            self.insertion_order.push_back(key.clone());
        }
        self.map.insert(key, value);
    }
}
