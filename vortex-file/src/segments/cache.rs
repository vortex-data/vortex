// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::{Arc, OnceLock};

use async_trait::async_trait;
use futures::FutureExt;
use moka::future::{Cache, CacheBuilder};
use moka::policy::EvictionPolicy;
use parking_lot::RwLock;
use rustc_hash::FxBuildHasher;
use vortex_array::serde::ArrayParts;
use vortex_array::{ArrayContext, ArrayRef};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_layout::segments::{ArrayCache, ArrayFuture, SegmentFuture, SegmentId, SegmentSource};
use vortex_metrics::{Counter, VortexMetrics};
use vortex_utils::aliases::dash_map::DashMap;

pub type EvictionCallback = Box<dyn Fn(Arc<SegmentId>) + Send + Sync>;

/// A cache for storing and retrieving individual segment data.
#[async_trait]
pub trait SegmentCache: Send + Sync {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>>;
    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()>;

    fn on_evict(&self, callback: EvictionCallback);
}

pub(crate) struct NoOpSegmentCache;

#[async_trait]
impl SegmentCache for NoOpSegmentCache {
    async fn get(&self, _id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(None)
    }

    async fn put(&self, _id: SegmentId, _buffer: ByteBuffer) -> VortexResult<()> {
        Ok(())
    }

    fn on_evict(&self, _callback: EvictionCallback) {}
}

/// A [`SegmentCache`] based around an in-memory Moka cache.
pub struct MokaSegmentCache {
    cache: Cache<SegmentId, ByteBuffer, FxBuildHasher>,
    on_evict_callbacks: Arc<RwLock<Vec<EvictionCallback>>>,
}

impl MokaSegmentCache {
    pub fn new(max_capacity_bytes: u64) -> Self {
        let on_evict_callbacks: Arc<RwLock<Vec<EvictionCallback>>> = Default::default();
        let callbacks = on_evict_callbacks.clone();

        let cache = CacheBuilder::new(max_capacity_bytes)
            .name("vortex-segment-cache")
            // Weight each segment by the number of bytes in the buffer.
            .weigher(|_, buffer: &ByteBuffer| {
                u32::try_from(buffer.len().min(u32::MAX as usize)).vortex_expect("must fit")
            })
            // We configure LFU (vs LRU) since the cache is mostly used when re-reading the
            // same file - it is _not_ used when reading the same segments during a single
            // scan.
            .eviction_policy(EvictionPolicy::tiny_lfu())
            .eviction_listener(move |key: Arc<SegmentId>, _value, cause| {
                if !cause.was_evicted() {
                    return;
                }
                for callback in callbacks.read().iter() {
                    callback(key.clone());
                }
            })
            .build_with_hasher(FxBuildHasher);

        Self {
            cache,
            on_evict_callbacks,
        }
    }
}

#[async_trait]
impl SegmentCache for MokaSegmentCache {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(self.cache.get(&id).await)
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.cache.insert(id, buffer).await;
        Ok(())
    }

    fn on_evict(&self, callback: EvictionCallback) {
        self.on_evict_callbacks.write().push(callback);
    }
}

/// Segment cache containing the initial read segments.
pub(crate) struct InitialReadSegmentCache {
    pub(crate) initial: DashMap<SegmentId, ByteBuffer>,
    pub(crate) fallback: Arc<dyn SegmentCache>,
}

#[async_trait]
impl SegmentCache for InitialReadSegmentCache {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        if let Some(buffer) = self.initial.get(&id) {
            return Ok(Some(buffer.clone()));
        }
        self.fallback.get(id).await
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.fallback.put(id, buffer).await
    }

    fn on_evict(&self, callback: EvictionCallback) {
        // we don't evict from initial
        self.fallback.on_evict(callback);
    }
}

pub struct SegmentCacheMetrics<C> {
    segment_cache: C,

    hits: Arc<Counter>,
    misses: Arc<Counter>,
    stores: Arc<Counter>,
}

impl<C: SegmentCache> SegmentCacheMetrics<C> {
    pub fn new(segment_cache: C, metrics: VortexMetrics) -> Self {
        Self {
            segment_cache,
            hits: metrics.counter("vortex.file.segments.cache.hits"),
            misses: metrics.counter("vortex.file.segments.cache.misses"),
            stores: metrics.counter("vortex.file.segments.cache.stores"),
        }
    }
}

#[async_trait]
impl<C: SegmentCache> SegmentCache for SegmentCacheMetrics<C> {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        let result = self.segment_cache.get(id).await?;
        if result.is_some() {
            self.hits.inc()
        } else {
            self.misses.inc()
        }
        Ok(result)
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.segment_cache.put(id, buffer).await?;
        self.stores.inc();
        Ok(())
    }

    fn on_evict(&self, callback: EvictionCallback) {
        self.segment_cache.on_evict(callback);
    }
}

pub struct SegmentCacheSourceAdapter {
    pub(crate) cache: Arc<dyn SegmentCache>,
    source: Arc<dyn SegmentSource>,
    array_cache: OnceLock<Arc<dyn ArrayCache>>,
}

impl SegmentCacheSourceAdapter {
    pub fn new(cache: Arc<dyn SegmentCache>, source: Arc<dyn SegmentSource>) -> Self {
        Self {
            cache,
            source,
            array_cache: OnceLock::new(),
        }
    }
}

impl SegmentSource for SegmentCacheSourceAdapter {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let cache = self.cache.clone();
        let delegate = self.source.request(id);

        async move {
            if let Ok(Some(segment)) = cache.get(id).await {
                log::debug!("Resolved segment {} from cache", id);
                return Ok(segment);
            }
            let result = delegate.await?;
            if let Err(e) = cache.put(id, result.clone()).await {
                log::warn!("Failed to store segment {} in cache: {}", id, e);
            }
            Ok(result)
        }
        .boxed()
    }

    fn array_cache(self: Arc<Self>) -> Option<Arc<dyn ArrayCache>> {
        let cache = self
            .array_cache
            .get_or_init(|| Arc::new(SharedArrayCache::new(self.clone())));
        Some(cache.clone())
    }
}

pub struct SharedArrayCache {
    arrays: DashMap<SegmentId, ArrayRef>,
    segment_source: Arc<SegmentCacheSourceAdapter>,
}

impl SharedArrayCache {
    pub fn new(segment_source: Arc<SegmentCacheSourceAdapter>) -> Self {
        let arrays: DashMap<SegmentId, ArrayRef> = Default::default();
        let arrays_clone = arrays.clone();
        segment_source
            .cache
            .on_evict(Box::new(move |id: Arc<SegmentId>| {
                arrays_clone.remove(&id);
            }));

        Self {
            arrays,
            segment_source,
        }
    }
}
impl ArrayCache for SharedArrayCache {
    fn get<'a>(
        &'a self,
        segment_id: SegmentId,
        ctx: ArrayContext,
        dtype: DType,
        len: usize,
    ) -> ArrayFuture<'a> {
        async move {
            if let Some(entry) = self.arrays.get(&segment_id) {
                return Ok(entry.value().clone());
            }
            // NOTE: this method doesn't lock, so duplicate inflight segment requests
            // for the same segment could happen
            let segment = self.segment_source.request(segment_id).await?;
            let array = ArrayParts::try_from(segment)?.decode(&ctx, &dtype, len)?;
            self.arrays.insert(segment_id, array.clone());
            Ok(array)
        }
        .boxed()
    }
}
