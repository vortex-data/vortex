use std::sync::RwLock;

use async_trait::async_trait;
use vortex_array::aliases::hash_map::HashMap;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{vortex_err, VortexResult};
use vortex_layout::segments::SegmentId;

/// A cache for storing and retrieving individual segment data.
#[async_trait]
pub trait SegmentCache: Send + Sync {
    async fn get(&self, id: SegmentId, alignment: Alignment) -> VortexResult<Option<ByteBuffer>>;
    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()>;
    async fn remove(&self, id: SegmentId) -> VortexResult<()>;
}

pub(crate) struct NoOpSegmentCache;

#[async_trait]
impl SegmentCache for NoOpSegmentCache {
    async fn get(&self, _id: SegmentId, _alignment: Alignment) -> VortexResult<Option<ByteBuffer>> {
        Ok(None)
    }

    async fn put(&self, _id: SegmentId, _buffer: ByteBuffer) -> VortexResult<()> {
        Ok(())
    }

    async fn remove(&self, _id: SegmentId) -> VortexResult<()> {
        Ok(())
    }
}

#[derive(Default)]
pub(crate) struct InMemorySegmentCache(RwLock<HashMap<SegmentId, ByteBuffer>>);

#[async_trait]
impl SegmentCache for InMemorySegmentCache {
    async fn get(&self, id: SegmentId, _alignment: Alignment) -> VortexResult<Option<ByteBuffer>> {
        Ok(self
            .0
            .read()
            .map_err(|_| vortex_err!("poisoned"))?
            .get(&id)
            .cloned())
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.0
            .write()
            .map_err(|_| vortex_err!("poisoned"))?
            .insert(id, buffer);
        Ok(())
    }

    async fn remove(&self, id: SegmentId) -> VortexResult<()> {
        self.0
            .write()
            .map_err(|_| vortex_err!("poisoned"))?
            .remove(&id);
        Ok(())
    }
}
