// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fs::File;
use std::sync::Arc;

use futures::FutureExt;
use futures::future;
use vortex_array::buffer::BufferHandle;
use vortex_array::memory::BufferAllocatorRef;
use vortex_io::std_file::try_read_exact_at_cached;
use vortex_layout::segments::SegmentFuture;
use vortex_layout::segments::SegmentId;
use vortex_layout::segments::SegmentSource;

use crate::SegmentSpec;

/// [`SegmentSource`] that serves segments of a local file straight from the OS page cache.
///
/// Each request first issues a non-blocking positional read (`preadv2(RWF_NOWAIT)` on Linux) of
/// the segment's byte range on the calling thread. If every byte is resident in the page cache the
/// request resolves immediately, without registering with the coalescing read driver, the
/// in-flight deduplication map or the blocking thread pool. Otherwise the request is forwarded to
/// the wrapped source, whose read also populates the page cache.
///
/// On platforms without non-blocking reads every request is forwarded.
pub struct PageCacheSegmentSource {
    file: Arc<File>,
    segments: Arc<[SegmentSpec]>,
    allocator: BufferAllocatorRef,
    fallback: Arc<dyn SegmentSource>,
}

impl PageCacheSegmentSource {
    /// Serve `segments` of `file` from the page cache, forwarding misses to `fallback`.
    pub fn new(
        file: Arc<File>,
        segments: Arc<[SegmentSpec]>,
        allocator: BufferAllocatorRef,
        fallback: Arc<dyn SegmentSource>,
    ) -> Self {
        Self {
            file,
            segments,
            allocator,
            fallback,
        }
    }

    fn try_read_cached(&self, id: SegmentId) -> Option<BufferHandle> {
        let spec = self.segments.get(*id as usize)?;
        let length = spec.length as usize;
        let mut buffer = self
            .allocator
            .with_capacity_aligned::<u8>(length, spec.alignment);
        // SAFETY: the buffer is only frozen once every byte has been read into it.
        unsafe { buffer.set_len(length) };
        match try_read_exact_at_cached(&self.file, buffer.as_mut_slice(), spec.offset) {
            Ok(true) => Some(BufferHandle::new_host(buffer.freeze())),
            Ok(false) => None,
            Err(e) => {
                tracing::debug!("Non-blocking read of segment {} failed: {}", id, e);
                None
            }
        }
    }
}

impl SegmentSource for PageCacheSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        match self.try_read_cached(id) {
            Some(buffer) => future::ready(Ok(buffer)).boxed(),
            None => self.fallback.request(id),
        }
    }
}

#[cfg(test)]
#[cfg(target_os = "linux")]
mod tests {
    use std::fs::File;
    use std::sync::Arc;

    use futures::FutureExt;
    use futures::future;
    use vortex_array::buffer::BufferHandle;
    use vortex_array::memory::BufferAllocatorRef;
    use vortex_buffer::Alignment;
    use vortex_buffer::ByteBuffer;
    use vortex_error::VortexResult;
    use vortex_layout::segments::SegmentFuture;
    use vortex_layout::segments::SegmentId;
    use vortex_layout::segments::SegmentSource;

    use super::PageCacheSegmentSource;
    use crate::SegmentSpec;

    struct ConstantSource(ByteBuffer);

    impl SegmentSource for ConstantSource {
        fn request(&self, _id: SegmentId) -> SegmentFuture {
            future::ready(Ok(BufferHandle::new_host(self.0.clone()))).boxed()
        }
    }

    #[tokio::test]
    async fn serves_resident_segments_and_forwards_misses() -> VortexResult<()> {
        let path = std::env::temp_dir().join(format!(
            "vortex-page-cache-segment-source-{}.bin",
            std::process::id()
        ));
        let bytes: Vec<u8> = (0..=255).collect();
        // Freshly written pages are resident in the page cache.
        std::fs::write(&path, &bytes)?;
        let file = Arc::new(File::open(&path)?);
        std::fs::remove_file(&path)?;

        let fallback_data = ByteBuffer::from(vec![9u8; 4]);
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
        let source = PageCacheSegmentSource::new(
            file,
            segments,
            BufferAllocatorRef::statically_allocated(),
            Arc::new(ConstantSource(fallback_data.clone())),
        );

        let hit = source.request(SegmentId::from(0)).await?.unwrap_host();
        assert_eq!(hit.as_slice(), &bytes[16..80]);
        assert!(hit.is_aligned(Alignment::new(8)));

        let miss = source.request(SegmentId::from(1)).await?.unwrap_host();
        assert_eq!(miss, fallback_data);
        Ok(())
    }
}
