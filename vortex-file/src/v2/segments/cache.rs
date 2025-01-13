use std::sync::RwLock;

use async_trait::async_trait;
use vortex_array::aliases::hash_map::HashMap;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{vortex_err, VortexExpect};
use vortex_layout::segments::SegmentId;

/// A cache for storing and retrieving individual segment data.
#[async_trait]
pub trait SegmentCache: Send + Sync {
    async fn get(&self, id: SegmentId, alignment: Alignment) -> Option<ByteBuffer>;
    async fn put(&self, id: SegmentId, buffer: ByteBuffer);
    async fn remove(&self, id: SegmentId);
}

pub(crate) struct NoOpSegmentCache;

#[async_trait]
impl SegmentCache for NoOpSegmentCache {
    async fn get(&self, _id: SegmentId, _alignment: Alignment) -> Option<ByteBuffer> {
        None
    }

    async fn put(&self, _id: SegmentId, _buffer: ByteBuffer) {}

    async fn remove(&self, _id: SegmentId) {}
}

#[derive(Default)]
pub(crate) struct InMemorySegmentCache(RwLock<HashMap<SegmentId, ByteBuffer>>);

#[async_trait]
impl SegmentCache for InMemorySegmentCache {
    async fn get(&self, id: SegmentId, _alignment: Alignment) -> Option<ByteBuffer> {
        self.0
            .read()
            .map_err(|_| vortex_err!("poisoned"))
            .vortex_expect("poisoned")
            .get(&id)
            .cloned()
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) {
        self.0
            .write()
            .map_err(|_| vortex_err!("poisoned"))
            .vortex_expect("poisoned")
            .insert(id, buffer);
    }

    async fn remove(&self, id: SegmentId) {
        self.0
            .write()
            .map_err(|_| vortex_err!("poisoned"))
            .vortex_expect("poisoned")
            .remove(&id);
    }
}
