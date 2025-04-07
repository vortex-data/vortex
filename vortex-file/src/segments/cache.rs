use async_trait::async_trait;
use moka::future::{Cache, CacheBuilder};
use moka::policy::EvictionPolicy;
use rustc_hash::FxBuildHasher;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexExpect, VortexResult};
use vortex_layout::segments::SegmentId;

/// A cache for storing and retrieving individual segment data.
#[async_trait]
pub trait SegmentCache: Send + Sync {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>>;
    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()>;
    async fn remove(&self, id: SegmentId) -> VortexResult<()>;
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

    async fn remove(&self, _id: SegmentId) -> VortexResult<()> {
        Ok(())
    }
}

pub(crate) struct InMemorySegmentCache(Cache<SegmentId, ByteBuffer, FxBuildHasher>);

impl InMemorySegmentCache {
    pub fn new(builder: CacheBuilder<SegmentId, ByteBuffer, Cache<SegmentId, ByteBuffer>>) -> Self {
        Self(
            builder
                // Weight each segment by the number of bytes in the buffer.
                .weigher(|_, buffer| {
                    u32::try_from(buffer.len().min(u32::MAX as usize)).vortex_expect("must fit")
                })
                // We configure LRU instead of LFU since we're likely to re-read segments between
                // filter and projection.
                .eviction_policy(EvictionPolicy::lru())
                .build_with_hasher(FxBuildHasher),
        )
    }
}

#[async_trait]
impl SegmentCache for InMemorySegmentCache {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(self.0.get(&id).await)
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.0.insert(id, buffer).await;
        Ok(())
    }

    async fn remove(&self, id: SegmentId) -> VortexResult<()> {
        self.0.invalidate(&id).await;
        Ok(())
    }
}
