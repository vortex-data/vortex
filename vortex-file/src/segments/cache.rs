use moka::policy::EvictionPolicy;
use moka::sync::{Cache, CacheBuilder};
use rustc_hash::FxBuildHasher;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexExpect, VortexResult};
use vortex_layout::segments::SegmentId;

/// A cache for storing and retrieving individual segment data.
pub trait SegmentCache: Send + Sync {
    fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>>;
    fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()>;
    fn remove(&self, id: SegmentId) -> VortexResult<()>;
}

pub(crate) struct NoOpSegmentCache;

impl SegmentCache for NoOpSegmentCache {
    fn get(&self, _id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(None)
    }

    fn put(&self, _id: SegmentId, _buffer: ByteBuffer) -> VortexResult<()> {
        Ok(())
    }

    fn remove(&self, _id: SegmentId) -> VortexResult<()> {
        Ok(())
    }
}

#[derive(Debug)]
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

impl SegmentCache for InMemorySegmentCache {
    fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(self.0.get(&id))
    }

    fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.0.insert(id, buffer);
        Ok(())
    }

    fn remove(&self, id: SegmentId) -> VortexResult<()> {
        self.0.invalidate(&id);
        Ok(())
    }
}
