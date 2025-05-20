use std::sync::Arc;

use async_trait::async_trait;
use dashmap::DashMap;
use futures::FutureExt;
use moka::future::{Cache, CacheBuilder};
use moka::policy::EvictionPolicy;
use rustc_hash::FxBuildHasher;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexExpect, VortexResult};
use vortex_layout::segments::{SegmentFuture, SegmentId, SegmentSource};
use vortex_metrics::{Counter, VortexMetrics};

/// A cache for storing and retrieving individual segment data.
#[async_trait]
pub trait SegmentCache: Send + Sync {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>>;
    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()>;
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
}

/// A [`SegmentCache`] based around an in-memory Moka cache.
pub struct MokaSegmentCache(Cache<SegmentId, ByteBuffer, FxBuildHasher>);

impl MokaSegmentCache {
    pub fn new(max_capacity_bytes: u64) -> Self {
        Self(
            CacheBuilder::new(max_capacity_bytes)
                .name("vortex-segment-cache")
                // Weight each segment by the number of bytes in the buffer.
                .weigher(|_, buffer: &ByteBuffer| {
                    u32::try_from(buffer.len().min(u32::MAX as usize)).vortex_expect("must fit")
                })
                // We configure LFU (vs LRU) since the cache is mostly used when re-reading the
                // same file - it is _not_ used when reading the same segments during a single
                // scan.
                .eviction_policy(EvictionPolicy::tiny_lfu())
                .build_with_hasher(FxBuildHasher),
        )
    }
}

#[async_trait]
impl SegmentCache for MokaSegmentCache {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(self.0.get(&id).await)
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.0.insert(id, buffer).await;
        Ok(())
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
}

pub struct SegmentCacheSourceAdapter {
    cache: Arc<dyn SegmentCache>,
    source: Arc<dyn SegmentSource>,
}

impl SegmentCacheSourceAdapter {
    pub fn new(cache: Arc<dyn SegmentCache>, source: Arc<dyn SegmentSource>) -> Self {
        Self { cache, source }
    }
}

impl SegmentSource for SegmentCacheSourceAdapter {
    fn request(&self, id: SegmentId, for_whom: &Arc<str>) -> SegmentFuture {
        let cache = self.cache.clone();
        let delegate = self.source.request(id, for_whom);
        let for_whom = for_whom.clone();

        async move {
            if let Ok(Some(segment)) = cache.get(id).await {
                log::debug!("Resolved segment {} for {} from cache", id, &for_whom);
                return Ok(segment);
            }
            let result = delegate.await?;
            if let Err(e) = cache.put(id, result.clone()).await {
                log::warn!(
                    "Failed to store segment {} for {} in cache: {}",
                    id,
                    &for_whom,
                    e
                );
            }
            Ok(result)
        }
        .boxed()
    }
}
