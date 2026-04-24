// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

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

/// A cache for storing and retrieving individual segment data.
#[async_trait]
pub trait SegmentCache: Send + Sync {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>>;
    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()>;
}

pub struct NoOpSegmentCache;

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
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        let result = self.segment_cache.get(id).await?;
        if result.is_some() {
            self.hits.add(1);
        } else {
            self.misses.add(1);
        }
        Ok(result)
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.segment_cache.put(id, buffer).await?;
        self.stores.add(1);
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
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let cache = Arc::clone(&self.cache);
        let delegate = self.source.request(id);

        async move {
            if let Ok(Some(segment)) = cache.get(id).await {
                tracing::debug!("Resolved segment {} from cache", id);
                return Ok(BufferHandle::new_host(segment));
            }
            let result = delegate.await?;
            // Cache only CPU buffers; device buffers are not cached.
            if let Some(buffer) = result.as_host_opt()
                && let Err(e) = cache.put(id, buffer.clone()).await
            {
                tracing::warn!("Failed to store segment {} in cache: {}", id, e);
            }
            Ok(result)
        }
        .boxed()
    }
}
