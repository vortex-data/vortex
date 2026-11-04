// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! EXPERIMENT: keep the segments covered by the footer's initial read across file opens.
//!
//! `VORTEX_FOOTER_SEGMENT_CACHE=1` enables a process-wide cache that only admits segments while a
//! file is being opened (i.e. those the initial read covered), so a later open that reuses a cached
//! footer still finds them. `VORTEX_INITIAL_READ_SIZE=<bytes>` overrides the initial read size at
//! every open site.

use std::sync::Arc;
use std::sync::LazyLock;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::Ordering;

use async_trait::async_trait;
use object_store::ObjectMeta;
use vortex::buffer::ByteBuffer;
use vortex::error::VortexResult;
use vortex::file::VortexOpenOptions;
use vortex::layout::segments::FileSegmentCache;
use vortex::layout::segments::MokaSegmentCache;
use vortex::layout::segments::SegmentCache;
use vortex::layout::segments::SegmentEviction;
use vortex::layout::segments::SegmentId;

static CACHE: LazyLock<Option<MokaSegmentCache>> = LazyLock::new(|| {
    std::env::var("VORTEX_FOOTER_SEGMENT_CACHE")
        .is_ok_and(|v| v == "1")
        .then(|| MokaSegmentCache::new(1 << 30, SegmentEviction::Lru))
});

static INITIAL_READ_SIZE: LazyLock<Option<usize>> = LazyLock::new(|| {
    std::env::var("VORTEX_INITIAL_READ_SIZE")
        .ok()
        .and_then(|v| v.parse().ok())
});

struct OpenPhaseCache {
    inner: FileSegmentCache,
    admit: Arc<AtomicBool>,
}

#[async_trait]
impl SegmentCache for OpenPhaseCache {
    async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
        self.inner.get(id).await
    }

    async fn put(&self, id: SegmentId, buffer: ByteBuffer) -> VortexResult<()> {
        if self.admit.load(Ordering::Relaxed) {
            self.inner.put(id, buffer).await?;
        }
        Ok(())
    }
}

/// Apply the experiment to `opts`. Call the returned closure once `open_read` returns so that scan
/// reads are no longer admitted.
pub(crate) fn configure(
    mut opts: VortexOpenOptions,
    meta: &ObjectMeta,
) -> (VortexOpenOptions, impl FnOnce()) {
    if let Some(size) = *INITIAL_READ_SIZE {
        opts = opts.with_initial_read_size(size);
    }
    let admit = Arc::new(AtomicBool::new(true));
    if let Some(cache) = CACHE.as_ref() {
        let id = format!("{}:{}:{:?}", meta.location, meta.size, meta.last_modified);
        opts = opts.with_segment_cache(Arc::new(OpenPhaseCache {
            inner: cache.for_file(id),
            admit: Arc::clone(&admit),
        }));
    }
    (opts, move || admit.store(false, Ordering::Relaxed))
}
