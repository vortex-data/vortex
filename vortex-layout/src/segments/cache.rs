// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt;
use futures::future;
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
    /// Return immediately available cached bytes, or `None` to use the asynchronous lookup.
    /// This must not perform IO. A miss still goes through [`Self::get`], which may resolve
    /// while the source's announced request waits to be polled.
    fn get_if_ready(&self, _id: SegmentId) -> Option<ByteBuffer> {
        None
    }

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

/// A [`SegmentCache`] based around an in-memory Moka cache.
pub struct MokaSegmentCache(Cache<SegmentId, ByteBuffer, FxBuildHasher>);

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
    fn get_if_ready(&self, id: SegmentId) -> Option<ByteBuffer> {
        self.0.get(&id).now_or_never().flatten()
    }

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
    fn get_if_ready(&self, id: SegmentId) -> Option<ByteBuffer> {
        let segment = self.segment_cache.get_if_ready(id)?;
        self.hits.add(1);
        Some(segment)
    }

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
        if let Some(segment) = self.cache.get_if_ready(id) {
            tracing::debug!("Resolved segment {} from cache", id);
            return future::ready(Ok(BufferHandle::new_host(segment))).boxed();
        }
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
    use std::sync::atomic::AtomicUsize;
    use std::sync::atomic::Ordering;

    use rstest::rstest;

    use super::*;

    struct TestCache {
        bytes: Option<ByteBuffer>,
        immediate: bool,
    }

    #[async_trait]
    impl SegmentCache for TestCache {
        fn get_if_ready(&self, _id: SegmentId) -> Option<ByteBuffer> {
            self.immediate.then(|| self.bytes.clone()).flatten()
        }

        async fn get(&self, _id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
            Ok(self.bytes.clone())
        }

        async fn put(&self, _id: SegmentId, _buffer: ByteBuffer) -> VortexResult<()> {
            Ok(())
        }
    }

    struct RecordingSource {
        bytes: ByteBuffer,
        requests: AtomicUsize,
        polls: Arc<AtomicUsize>,
    }

    impl SegmentSource for RecordingSource {
        fn request(&self, _id: SegmentId) -> SegmentFuture {
            self.requests.fetch_add(1, Ordering::Relaxed);
            let polls = Arc::clone(&self.polls);
            let bytes = self.bytes.clone();
            async move {
                polls.fetch_add(1, Ordering::Relaxed);
                Ok(BufferHandle::new_host(bytes))
            }
            .boxed()
        }
    }

    #[rstest]
    #[case::immediate_hit(true, true, 0, 0)]
    #[case::asynchronous_hit(true, false, 1, 0)]
    #[case::miss(false, true, 1, 1)]
    #[tokio::test]
    async fn cache_hits_skip_reads_and_misses_keep_early_announcements(
        #[case] cached: bool,
        #[case] immediate: bool,
        #[case] requests: usize,
        #[case] polls: usize,
    ) -> VortexResult<()> {
        let bytes = ByteBuffer::from(vec![1, 2, 3, 4]);
        let source = Arc::new(RecordingSource {
            bytes: bytes.clone(),
            requests: AtomicUsize::new(0),
            polls: Arc::default(),
        });
        let adapter = SegmentCacheSourceAdapter::new(
            Arc::new(TestCache {
                bytes: cached.then(|| bytes.clone()),
                immediate,
            }),
            Arc::<RecordingSource>::clone(&source),
        );
        let request = adapter.request(SegmentId::from(0));
        assert_eq!(source.requests.load(Ordering::Relaxed), requests);
        assert_eq!(source.polls.load(Ordering::Relaxed), 0);
        assert_eq!(request.await?.try_into_host()?.await?, bytes);
        assert_eq!(source.polls.load(Ordering::Relaxed), polls);
        Ok(())
    }
}
