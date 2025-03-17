use std::sync::Arc;

use futures::future::BoxFuture;
use futures::{FutureExt, Stream, stream};
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexResult, vortex_err};
use vortex_layout::scan::ScanDriver;
use vortex_layout::segments::{AsyncSegmentReader, SegmentId, SegmentStream};
use vortex_metrics::VortexMetrics;

use crate::{FileType, Footer, Segment, VortexFileDyn, VortexFileRef, VortexOpenOptions};

/// A Vortex file that is backed by an in-memory buffer.
///
/// This type of file reader performs no coalescing or other clever orchestration, simply
/// zero-copy slicing the segments from the buffer.
#[derive(Clone)]
pub struct InMemoryVortexFile {
    buffer: ByteBuffer,
    footer: Footer,
    metrics: VortexMetrics,
}

impl VortexOpenOptions<InMemoryVortexFile> {
    /// Open an in-memory file contained in the provided buffer.
    pub fn in_memory<B: Into<ByteBuffer>>(buffer: B) -> Self {
        Self::new(buffer.into(), ())
    }
}

impl FileType for InMemoryVortexFile {
    type Options = ();
    type Read = ByteBuffer;

    fn open(options: VortexOpenOptions<Self>, footer: Footer) -> VortexResult<VortexFileRef> {
        Ok(Arc::new(InMemoryVortexFile {
            buffer: options.read,
            footer,
            metrics: options.metrics,
        }))
    }
}

impl VortexFileDyn for InMemoryVortexFile {
    fn footer(&self) -> &Footer {
        &self.footer
    }

    fn metrics(&self) -> &VortexMetrics {
        &self.metrics
    }

    fn segment_reader(&self) -> Arc<dyn AsyncSegmentReader> {
        Arc::new(self.clone())
    }
}

impl ScanDriver for InMemoryVortexFile {
    fn segment_reader(&self) -> Arc<dyn AsyncSegmentReader> {
        Arc::new(self.clone())
    }

    fn io_stream(self, _segments: SegmentStream) -> impl Stream<Item = VortexResult<()>> {
        stream::repeat_with(|| Ok(()))
    }
}

impl AsyncSegmentReader for InMemoryVortexFile {
    fn get(&self, id: SegmentId) -> BoxFuture<'static, VortexResult<ByteBuffer>> {
        let segment_map = self.footer.segment_map().clone();
        let buffer = self.buffer.clone();

        async move {
            let segment: &Segment = segment_map
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
