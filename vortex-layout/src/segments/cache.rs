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

/// Cache for individual segment byte buffers.
///
/// Caches are optional and operate above a [`SegmentSource`]. They should only store host buffers:
/// device buffers and other non-host handles should be passed through uncached.
#[async_trait]
pub trait SegmentCache: Send + Sync {
    /// Return a cached segment, or `None` on cache miss.
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>>;
    /// Store a segment in the cache.
    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()>;
}

/// Segment cache implementation that never stores anything.
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

/// An in-memory Moka cache of segments, capped by total buffer bytes, that any number of files can
/// share.
///
/// A [`SegmentId`] is unique only within one file, so files use the cache through a
/// [`FileSegmentCache`] from [`Self::for_file`], which adds a file key. Files opened with the
/// same key share entries, so a key must identify the file's contents, not only its location.
/// Opening the same file again with the same key reuses the segments an earlier open read.
#[derive(Clone)]
pub struct MokaSegmentCache(Cache<(Arc<str>, SegmentId), ByteBuffer, FxBuildHasher>);

impl MokaSegmentCache {
    /// Construct a Moka-backed cache capped by total buffer bytes.
    pub fn new(max_capacity_bytes: u64) -> Self {
        Self(
            CacheBuilder::new(max_capacity_bytes)
                .name("vortex-segment-cache")
                // Weight each segment by the number of bytes in the buffer.
                .weigher(|_, buffer: &ByteBuffer| {
                    u32::try_from(buffer.len().min(u32::MAX as usize)).vortex_expect("must fit")
                })
                // Readers of many files move from file to file, so the segments of a newly opened
                // file must displace those of older files. TinyLFU admission would instead reject
                // new segments while older ones have higher counts.
                .eviction_policy(EvictionPolicy::lru())
                .build_with_hasher(FxBuildHasher),
        )
    }

    /// The view of this cache for the file identified by `key`.
    pub fn for_file(&self, key: impl Into<Arc<str>>) -> FileSegmentCache {
        FileSegmentCache {
            cache: self.clone(),
            key: key.into(),
        }
    }

    /// The total bytes of the cached segments.
    ///
    /// Recent inserts and evictions may not be counted yet; call [`Self::run_pending_tasks`] first
    /// for an exact value.
    pub fn weighted_size(&self) -> u64 {
        self.0.weighted_size()
    }

    /// The number of cached segments, with the same caveat as [`Self::weighted_size`].
    pub fn entry_count(&self) -> u64 {
        self.0.entry_count()
    }

    /// Apply pending inserts and evictions, so that the counts are exact.
    pub async fn run_pending_tasks(&self) {
        self.0.run_pending_tasks().await;
    }

    /// Remove every cached segment.
    pub fn invalidate_all(&self) {
        self.0.invalidate_all();
    }
}

/// One file's view of a [`MokaSegmentCache`]; see [`MokaSegmentCache::for_file`].
#[derive(Clone)]
pub struct FileSegmentCache {
    cache: MokaSegmentCache,
    key: Arc<str>,
}

#[async_trait]
impl SegmentCache for FileSegmentCache {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(self.cache.0.get(&(Arc::clone(&self.key), id)).await)
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.cache
            .0
            .insert((Arc::clone(&self.key), id), buffer)
            .await;
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
    /// Wrap a segment cache and record hit/miss/store metrics with the supplied labels.
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

/// [`SegmentSource`] wrapper that consults a [`SegmentCache`] before the underlying source.
pub struct SegmentCacheSourceAdapter {
    cache: Arc<dyn SegmentCache>,
    source: Arc<dyn SegmentSource>,
}

impl SegmentCacheSourceAdapter {
    /// Construct a cache-fronted source.
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

#[cfg(test)]
mod tests {
    use vortex_buffer::ByteBuffer;

    use super::*;

    #[tokio::test]
    async fn shared_cache_separates_files() -> VortexResult<()> {
        let shared = MokaSegmentCache::new(1 << 20);
        let a = shared.for_file("a");
        let b = shared.for_file("b");
        let id = SegmentId::from(0);

        a.put(id, ByteBuffer::copy_from(b"from a")).await?;
        assert_eq!(a.get(id).await?.as_deref(), Some(b"from a".as_slice()));
        assert!(b.get(id).await?.is_none());

        // A later view with the same key sees what the earlier one stored.
        let a_again = shared.for_file("a");
        assert_eq!(
            a_again.get(id).await?.as_deref(),
            Some(b"from a".as_slice())
        );
        Ok(())
    }

    #[tokio::test]
    async fn shared_cache_is_capped_by_bytes() -> VortexResult<()> {
        let shared = MokaSegmentCache::new(1000);
        let file = shared.for_file("file");
        for i in 0..10u32 {
            file.put(SegmentId::from(i), ByteBuffer::copy_from(vec![0u8; 400]))
                .await?;
        }

        shared.run_pending_tasks().await;
        assert!(shared.weighted_size() <= 1000);
        assert!(shared.entry_count() <= 2);

        // The most recently stored segment survives under LRU eviction.
        assert!(file.get(SegmentId::from(9)).await?.is_some());
        Ok(())
    }
}
