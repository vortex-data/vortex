// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! [`PointSession`] — the caching point-fn dispatcher.

use std::any::Any;
use std::collections::VecDeque;
use std::sync::Arc;

use vortex_error::VortexResult;
use vortex_utils::aliases::hash_map::HashMap;

use crate::ArrayRef;
use crate::ExecutionCtx;
use crate::point_fn::BlockKey;
use crate::point_fn::PointDispatch;
use crate::point_fn::dispatch::CacheArrayId;
use crate::point_fn::dispatch_table;
use crate::scalar::Scalar;
use crate::search_sorted::SearchResult;
use crate::search_sorted::SearchSortedSide;

/// Default capacity for the per-session scalar cache.
pub const DEFAULT_SCALAR_CACHE_CAPACITY: usize = 64;

/// Default capacity for the per-session decoded-block cache.
pub const DEFAULT_BLOCK_CACHE_CAPACITY: usize = 8;

/// A caching point-fn dispatcher. Holds a bounded scalar cache and a bounded
/// decoded-block cache that persist for the lifetime of the session.
///
/// Construct via [`ArrayRef::point_session`](crate::ArrayRef::point_session) or via
/// [`PointSession::new`]. Hold the session across multiple point-fn calls to share
/// block decode work (Pco/Fsst/Delta/Zstd) and scalar lookups.
///
/// ## Eviction policy
///
/// Phase 1 uses FIFO eviction with a fixed capacity. A future change may upgrade
/// to LRU; the public API does not assume a specific policy.
pub struct PointSession<'a> {
    ctx: &'a mut ExecutionCtx,
    scalar_cache: BoundedFifo<(CacheArrayId, usize), Scalar>,
    block_cache: BoundedFifo<(CacheArrayId, BlockKey), Arc<dyn Any + Send + Sync>>,
}

impl<'a> PointSession<'a> {
    /// Construct a session with default cache capacities.
    pub fn new(ctx: &'a mut ExecutionCtx) -> Self {
        Self::with_capacities(
            ctx,
            DEFAULT_SCALAR_CACHE_CAPACITY,
            DEFAULT_BLOCK_CACHE_CAPACITY,
        )
    }

    /// Construct a session with explicit cache capacities.
    pub fn with_capacities(
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

    /// Number of entries currently in the scalar cache. Exposed for tests/benches.
    pub fn scalar_cache_len(&self) -> usize {
        self.scalar_cache.len()
    }

    /// Number of entries currently in the block cache. Exposed for tests/benches.
    pub fn block_cache_len(&self) -> usize {
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
        // Route via the dispatch table so view encodings (Slice today; Dict /
        // RunEnd / Chunked / etc. in later phases) push down through children.
        let v = dispatch_table::dispatch_scalar_at(arr, idx, self)?;
        self.scalar_cache.put(key, v.clone());
        Ok(v)
    }

    fn search_sorted(
        &mut self,
        arr: &ArrayRef,
        value: &Scalar,
        side: SearchSortedSide,
    ) -> VortexResult<SearchResult> {
        dispatch_table::dispatch_search_sorted(arr, value, side, self)
    }

    fn cached_block<B, F>(&mut self, key: (CacheArrayId, BlockKey), decode: F) -> VortexResult<B>
    where
        B: Clone + Send + Sync + 'static,
        F: FnOnce() -> VortexResult<B>,
    {
        if let Some(any) = self.block_cache.get(&key)
            && let Some(b) = any.downcast_ref::<B>()
        {
            return Ok(b.clone());
        }
        // Either cache miss, or type mismatch on the same key (very unusual; two
        // encodings would have to use the same `BlockKey`). Fall through and decode.
        let b = decode()?;
        let stored: Arc<dyn Any + Send + Sync> = Arc::new(b.clone());
        self.block_cache.put(key, stored);
        Ok(b)
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
