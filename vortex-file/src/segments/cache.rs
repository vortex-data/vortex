// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_layout::segments::SegmentCache;
use vortex_layout::segments::SegmentId;
use vortex_utils::aliases::hash_map::HashMap;

/// Segment cache containing the initial read segments.
pub struct InitialReadSegmentCache {
    /// Segments that were already covered by the footer initial read.
    pub initial: HashMap<SegmentId, ByteBuffer>,
    /// Delegate cache used for all misses and stores.
    pub fallback: Arc<dyn SegmentCache>,
}

#[async_trait]
impl SegmentCache for InitialReadSegmentCache {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        if let Some(buffer) = self.initial.get(&id) {
            return Ok(Some(buffer.clone()));
        }
        self.fallback.get(id).await
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.fallback.put(id, buffer).await
    }
}

/// The bit that [`PlaintextSegmentCache`] sets in the segment ids it gives to its cache.
const PLAINTEXT_ID_BIT: u32 = 1 << 31;

/// A view of a [`SegmentCache`] that keeps the plaintext of an encrypted file apart from its
/// ciphertext.
///
/// The cache keeps the ciphertext of a segment under the segment id. This view keeps the
/// plaintext under the segment id with [`PLAINTEXT_ID_BIT`] set. Thus, opens of one file with
/// different cache modes can share a cache and never read the entries of the other mode. Segment
/// ids that already have the bit set are not cached.
pub(crate) struct PlaintextSegmentCache {
    inner: Arc<dyn SegmentCache>,
}

impl PlaintextSegmentCache {
    pub(crate) fn new(inner: Arc<dyn SegmentCache>) -> Self {
        Self { inner }
    }

    /// The id that the plaintext of segment `id` uses in the cache, or `None` if it has none.
    fn plaintext_id(id: SegmentId) -> Option<SegmentId> {
        (*id & PLAINTEXT_ID_BIT == 0).then(|| SegmentId::from(*id | PLAINTEXT_ID_BIT))
    }
}

#[async_trait]
impl SegmentCache for PlaintextSegmentCache {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        match Self::plaintext_id(id) {
            Some(id) => self.inner.get(id).await,
            None => Ok(None),
        }
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        match Self::plaintext_id(id) {
            Some(id) => self.inner.put(id, buffer).await,
            None => Ok(()),
        }
    }
}
