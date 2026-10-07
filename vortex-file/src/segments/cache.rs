// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_layout::segments::SegmentCache;
use vortex_layout::segments::SegmentId;
use vortex_utils::aliases::hash_map::HashMap;

#[cfg(not(target_arch = "wasm32"))]
pub use page_cache::*;

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

#[cfg(not(target_arch = "wasm32"))]
mod page_cache {
    use std::fs::File;
    use std::sync::Arc;

    use async_trait::async_trait;
    use vortex_array::memory::BufferAllocatorRef;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_io::std_file::try_read_exact_at_cached;
    use vortex_layout::segments::SegmentCache;
    use vortex_layout::segments::SegmentId;

    use crate::SegmentSpec;

    /// Segment cache that serves segments from the OS page cache of a local file.
    ///
    /// Lookups issue a non-blocking positional read (`preadv2(RWF_NOWAIT)` on Linux) for the
    /// segment's byte range. If every byte is already resident in the page cache the segment is
    /// returned directly from the calling task, skipping the coalescing read path and the blocking
    /// thread pool. Otherwise the lookup is a miss and the regular read path fetches the segment,
    /// which also populates the page cache. Stores are no-ops since the OS owns eviction.
    ///
    /// On platforms without non-blocking reads every lookup is a miss.
    pub struct PageCacheSegmentCache {
        file: Arc<File>,
        segments: Arc<[SegmentSpec]>,
        allocator: BufferAllocatorRef,
    }

    impl PageCacheSegmentCache {
        /// Create a page cache backed segment cache for `file`, whose segments are `segments`.
        pub fn new(
            file: Arc<File>,
            segments: Arc<[SegmentSpec]>,
            allocator: BufferAllocatorRef,
        ) -> Self {
            Self {
                file,
                segments,
                allocator,
            }
        }
    }

    #[async_trait]
    impl SegmentCache for PageCacheSegmentCache {
        async fn get(&self, id: SegmentId) -> VortexResult<Option<ByteBuffer>> {
            let Some(spec) = self.segments.get(*id as usize) else {
                return Ok(None);
            };
            let length = spec.length as usize;
            let mut buffer = self
                .allocator
                .with_capacity_aligned::<u8>(length, spec.alignment);
            // SAFETY: the buffer is only frozen once every byte has been read into it.
            unsafe { buffer.set_len(length) };
            if try_read_exact_at_cached(&self.file, buffer.as_mut_slice(), spec.offset)? {
                Ok(Some(buffer.freeze()))
            } else {
                Ok(None)
            }
        }

        async fn put(&self, _id: SegmentId, _buffer: ByteBuffer) -> VortexResult<()> {
            Ok(())
        }
    }

    #[cfg(all(test, target_os = "linux"))]
    mod tests {
        use std::fs::File;
        use std::sync::Arc;

        use vortex_array::memory::BufferAllocatorRef;
        use vortex_buffer::Alignment;
        use vortex_error::VortexResult;
        use vortex_layout::segments::SegmentCache;
        use vortex_layout::segments::SegmentId;

        use super::PageCacheSegmentCache;
        use crate::SegmentSpec;

        #[tokio::test]
        async fn reads_resident_segments() -> VortexResult<()> {
            let path = std::env::temp_dir().join(format!(
                "vortex-page-cache-segment-cache-{}.bin",
                std::process::id()
            ));
            let bytes: Vec<u8> = (0..=255).collect();
            // Freshly written pages are resident in the page cache.
            std::fs::write(&path, &bytes)?;
            let file = Arc::new(File::open(&path)?);
            std::fs::remove_file(&path)?;

            let segments: Arc<[SegmentSpec]> = Arc::new([
                SegmentSpec {
                    offset: 16,
                    length: 64,
                    alignment: Alignment::new(8),
                },
                // Extends past the end of the file.
                SegmentSpec {
                    offset: 250,
                    length: 64,
                    alignment: Alignment::none(),
                },
            ]);
            let cache = PageCacheSegmentCache::new(
                file,
                segments,
                BufferAllocatorRef::statically_allocated(),
            );

            let hit = cache.get(SegmentId::from(0)).await?;
            assert_eq!(hit.as_ref().map(|b| b.as_slice()), Some(&bytes[16..80]));
            assert!(hit.is_some_and(|b| b.is_aligned(Alignment::new(8))));
            assert!(cache.get(SegmentId::from(1)).await?.is_none());
            assert!(cache.get(SegmentId::from(2)).await?.is_none());
            Ok(())
        }
    }
}
