use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexResult, vortex_err};
use vortex_layout::segments::{SegmentId, SegmentReader};

use crate::{FileType, Footer, SegmentSpec, VortexFile, VortexOpenOptions};

/// A Vortex file that is backed by an in-memory buffer.
///
/// This type of file reader performs no coalescing or other clever orchestration, simply
/// zero-copy slicing the segments from the buffer.
pub struct InMemoryVortexFile;

impl FileType for InMemoryVortexFile {
    type Options = ();
}

impl VortexOpenOptions<InMemoryVortexFile> {
    /// Open an in-memory file contained in the provided buffer.
    pub fn in_memory() -> Self {
        Self::new(())
    }

    /// Open an in-memory file contained in the provided buffer.
    pub async fn open<B: Into<ByteBuffer>>(self, buffer: B) -> VortexResult<VortexFile> {
        let buffer = buffer.into();
        let footer = self.read_footer(&buffer).await?;
        Ok(VortexFile {
            footer: footer.clone(),
            segment_cache: self.segment_cache,
            io_driver: None,
            metrics: self.metrics,
        })
    }
}

struct InMemorySegmentReader {
    buffer: ByteBuffer,
    footer: Footer,
}

impl SegmentReader for InMemorySegmentReader {
    fn get(
        &self,
        id: SegmentId,
        _for_whom: &Arc<str>,
    ) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let segment_map = self.footer.segment_map().clone();
        let buffer = self.buffer.clone();

        async move {
            let segment: &SegmentSpec = segment_map
                .get(*id as usize)
                .ok_or_else(|| vortex_err!("segment not found"))?;

            let start =
                usize::try_from(segment.offset).map_err(|_| vortex_err!("offset too large"))?;
            let end = start + segment.length as usize;

            Ok(buffer.slice(start..end))
        }
        .boxed()
    }
}
