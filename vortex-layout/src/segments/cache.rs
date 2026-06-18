// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use futures::FutureExt;
use moka::future::Cache;
use moka::future::CacheBuilder;
use moka::policy::EvictionPolicy;
use rustc_hash::FxBuildHasher;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_metrics::Counter;
use vortex_metrics::Label;
use vortex_metrics::MetricBuilder;
use vortex_metrics::MetricsRegistry;

use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSource;

static NEXT_SEGMENT_CACHE_SOURCE_ID: AtomicU64 = AtomicU64::new(0);

/// Source namespace for segment-cache keys.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SegmentCacheSourceId(u64);

impl SegmentCacheSourceId {
    /// Allocate a unique source namespace for one opened segment source.
    pub fn unique() -> Self {
        Self(NEXT_SEGMENT_CACHE_SOURCE_ID.fetch_add(1, Ordering::Relaxed))
    }

    /// Return the integer value of this source namespace.
    pub fn get(self) -> u64 {
        self.0
    }
}

/// Source-scoped key for a cached segment payload.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct SegmentCacheKey {
    /// Source namespace for this segment.
    pub source: SegmentCacheSourceId,
    /// Logical segment id within the source.
    pub segment_id: SegmentId,
}

impl SegmentCacheKey {
    /// Create a segment-cache key.
    pub fn new(source: SegmentCacheSourceId, segment_id: SegmentId) -> Self {
        Self { source, segment_id }
    }
}

/// A cache for storing and retrieving individual segment data.
#[async_trait]
pub trait SegmentCache: Send + Sync {
    async fn get(&self, key: SegmentCacheKey) -> VortexResult<Option<ByteBuffer>>;
    async fn put(&self, key: SegmentCacheKey, buffer: ByteBuffer) -> VortexResult<()>;
}

pub struct NoOpSegmentCache;

#[async_trait]
impl SegmentCache for NoOpSegmentCache {
    async fn get(&self, _key: SegmentCacheKey) -> VortexResult<Option<ByteBuffer>> {
        Ok(None)
    }

    async fn put(&self, _key: SegmentCacheKey, _buffer: ByteBuffer) -> VortexResult<()> {
        Ok(())
    }
}

/// A [`SegmentCache`] based around an in-memory Moka cache.
pub struct MokaSegmentCache(Cache<SegmentCacheKey, ByteBuffer, FxBuildHasher>);

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
    async fn get(&self, key: SegmentCacheKey) -> VortexResult<Option<ByteBuffer>> {
        Ok(self.0.get(&key).await)
    }

    async fn put(&self, key: SegmentCacheKey, buffer: ByteBuffer) -> VortexResult<()> {
        self.0.insert(key, buffer).await;
        Ok(())
    }
}

/// Wrapper for [`SegmentCache`] that tracks its hit rate.
pub struct InstrumentedSegmentCache<C> {
    segment_cache: C,

    hits: Counter,
    misses: Counter,
    stores: Counter,
}

impl<C: SegmentCache> InstrumentedSegmentCache<C> {
    pub fn new(
        segment_cache: C,
        metrics_registry: &dyn MetricsRegistry,
        labels: Vec<Label>,
    ) -> Self {
        Self {
            segment_cache,
            hits: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("vortex.file.segments.cache.hits"),
            misses: MetricBuilder::new(metrics_registry)
                .add_labels(labels.clone())
                .counter("vortex.file.segments.cache.misses"),
            stores: MetricBuilder::new(metrics_registry)
                .add_labels(labels)
                .counter("vortex.file.segments.cache.stores"),
        }
    }
}

#[async_trait]
impl<C: SegmentCache> SegmentCache for InstrumentedSegmentCache<C> {
    async fn get(&self, key: SegmentCacheKey) -> VortexResult<Option<ByteBuffer>> {
        let result = self.segment_cache.get(key).await?;
        if result.is_some() {
            self.hits.add(1);
        } else {
            self.misses.add(1);
        }
        Ok(result)
    }

    async fn put(&self, key: SegmentCacheKey, buffer: ByteBuffer) -> VortexResult<()> {
        self.segment_cache.put(key, buffer).await?;
        self.stores.add(1);
        Ok(())
    }
}

pub struct SegmentCacheSourceAdapter {
    source_id: SegmentCacheSourceId,
    cache: Arc<dyn SegmentCache>,
    source: Arc<dyn SegmentSource>,
}

impl SegmentCacheSourceAdapter {
    /// Create a cache adapter with a unique source namespace.
    pub fn new(cache: Arc<dyn SegmentCache>, source: Arc<dyn SegmentSource>) -> Self {
        Self {
            source_id: SegmentCacheSourceId::unique(),
            cache,
            source,
        }
    }

    /// Create a cache adapter with an explicit source namespace.
    ///
    /// This is where a future stable file/object identity can be threaded in to reuse segment
    /// cache entries across independently opened instances of the same source.
    pub fn with_source_id(
        source_id: SegmentCacheSourceId,
        cache: Arc<dyn SegmentCache>,
        source: Arc<dyn SegmentSource>,
    ) -> Self {
        Self {
            source_id,
            cache,
            source,
        }
    }
}

impl SegmentSource for SegmentCacheSourceAdapter {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let key = SegmentCacheKey::new(self.source_id, id);
        let cache = Arc::clone(&self.cache);
        let delegate = self.source.request(id);

        async move {
            if let Ok(Some(segment)) = cache.get(key).await {
                tracing::debug!("Resolved segment {} from cache", id);
                return Ok(BufferHandle::new_host(segment));
            }
            let result = delegate.await?;
            // Cache only CPU buffers; device buffers are not cached.
            if let Some(buffer) = result.as_host_opt()
                && let Err(e) = cache.put(key, buffer.clone()).await
            {
                tracing::warn!("Failed to store segment {} in cache: {}", id, e);
            }
            Ok(result)
        }
        .boxed()
    }
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;

    use super::*;

    #[test]
    fn cache_key_is_source_scoped() -> VortexResult<()> {
        let cache = MokaSegmentCache::new(1024);
        let segment = SegmentId::from(0);
        let source_a = SegmentCacheSourceId::unique();
        let source_b = SegmentCacheSourceId::unique();
        let key_a = SegmentCacheKey::new(source_a, segment);
        let key_b = SegmentCacheKey::new(source_b, segment);

        block_on(cache.put(key_a, ByteBuffer::from(vec![1, 2, 3])))?;

        assert!(block_on(cache.get(key_a))?.is_some());
        assert!(block_on(cache.get(key_b))?.is_none());

        Ok(())
    }
}
