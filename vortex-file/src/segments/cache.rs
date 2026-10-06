// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::env;
use std::sync::Arc;
use std::sync::LazyLock;

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

/// Environment variable that sizes the process-wide in-memory segment cache, in bytes.
///
/// When set to a positive integer, files opened without an explicit [`SegmentCache`] share one
/// in-memory cache keyed by file URI and segment id. This lets repeated scans of the same files
/// skip IO entirely, which benchmarks use to measure in-memory query performance.
pub(crate) const SEGMENT_CACHE_BYTES_ENV: &str = "VORTEX_SEGMENT_CACHE_BYTES";

type GlobalCache = moka::sync::Cache<(Arc<str>, SegmentId), ByteBuffer>;

static GLOBAL_SEGMENT_CACHE: LazyLock<Option<GlobalCache>> = LazyLock::new(|| {
    let capacity = env::var(SEGMENT_CACHE_BYTES_ENV)
        .ok()?
        .trim()
        .parse::<u64>()
        .ok()
        .filter(|&bytes| bytes > 0)?;
    tracing::info!("Enabling global {capacity} byte segment cache from {SEGMENT_CACHE_BYTES_ENV}");
    Some(
        moka::sync::Cache::builder()
            .name("vortex-global-segment-cache")
            .max_capacity(capacity)
            .weigher(|_, buffer: &ByteBuffer| u32::try_from(buffer.len()).unwrap_or(u32::MAX))
            .build(),
    )
});

/// Segment cache scoped to one file within the process-wide cache.
struct GlobalFileSegmentCache {
    uri: Arc<str>,
    cache: &'static GlobalCache,
}

#[async_trait]
impl SegmentCache for GlobalFileSegmentCache {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        Ok(self.cache.get(&(Arc::clone(&self.uri), id)))
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        self.cache.insert((Arc::clone(&self.uri), id), buffer);
        Ok(())
    }
}

/// Return the process-wide segment cache for the file at `uri`, if [`SEGMENT_CACHE_BYTES_ENV`]
/// enables it.
pub(crate) fn global_segment_cache(uri: &Arc<str>) -> Option<Arc<dyn SegmentCache>> {
    let cache = GLOBAL_SEGMENT_CACHE.as_ref()?;
    Some(Arc::new(GlobalFileSegmentCache {
        uri: Arc::clone(uri),
        cache,
    }))
}
