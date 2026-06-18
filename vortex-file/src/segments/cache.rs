// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_layout::segments::SegmentCache;
use vortex_layout::segments::SegmentCacheKey;
use vortex_layout::segments::SegmentId;
use vortex_utils::aliases::hash_map::HashMap;

/// Segment cache containing the initial read segments.
pub struct InitialReadSegmentCache {
    pub initial: RwLock<HashMap<SegmentId, ByteBuffer>>,
    pub fallback: Arc<dyn SegmentCache>,
}

#[async_trait]
impl SegmentCache for InitialReadSegmentCache {
    async fn get(&self, key: SegmentCacheKey) -> VortexResult<Option<ByteBuffer>> {
        if let Some(buffer) = self.initial.read().get(&key.segment_id) {
            return Ok(Some(buffer.clone()));
        }
        self.fallback.get(key).await
    }

    async fn put(&self, key: SegmentCacheKey, buffer: ByteBuffer) -> VortexResult<()> {
        self.fallback.put(key, buffer).await
    }
}
