use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexResult, vortex_err};
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};

use crate::{FileType, Footer, SegmentSpec, VortexFile, VortexOpenOptions};

/// A Vortex file that is backed by an in-memory buffer.
///
/// This type of file reader performs no coalescing or other clever orchestration, simply
/// zero-copy slicing the segments from the buffer.
pub struct InMemoryVortexFile;

impl VortexOpenOptions<InMemoryVortexFile> {
    /// Open an in-memory file contained in the provided buffer.
    pub fn in_memory<B: Into<ByteBuffer>>(buffer: B) -> Self {
        Self::new(buffer.into(), ())
    }
}

impl FileType for InMemoryVortexFile {
    type Options = ();
    type Read = ByteBuffer;

    fn open(options: VortexOpenOptions<Self>, footer: Footer) -> VortexResult<VortexFile> {
        Ok(VortexFile {
            footer: footer.clone(),
            segment_reader: Arc::new(InMemorySegmentReader {
                buffer: options.read,
                footer,
            }),
            metrics: options.metrics,
        })
    }
}

struct InMemorySegmentReader {
    buffer: ByteBuffer,
    footer: Footer,
}

impl AsyncSegmentReader for InMemorySegmentReader {
    fn get(&self, id: SegmentId) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
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
