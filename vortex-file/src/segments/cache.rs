// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use parking_lot::RwLock;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_layout::segments::{SegmentCache, SegmentId};
use vortex_utils::aliases::hash_map::HashMap;

/// Segment cache containing the initial read segments.
pub struct InitialReadSegmentCache {
    pub initial: RwLock<HashMap<SegmentId, ByteBuffer>>,
    pub fallback: Arc<dyn SegmentCache>,
}

#[async_trait]
impl SegmentCache for InitialReadSegmentCache {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        if let Some(buffer) = self.initial.read().get(&id) {
            return Ok(Some(buffer.clone()));
        }
        self.fallback.get(id).await
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.fallback.put(id, buffer).await
    }
}
