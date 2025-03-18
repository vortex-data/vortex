use std::sync::Arc;

use async_trait::async_trait;
use futures::{Stream, stream};
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexResult, vortex_err};
use vortex_layout::scan::ScanDriver;
use vortex_layout::segments::{AsyncSegmentReader, SegmentId, SegmentStream};
use vortex_metrics::VortexMetrics;

use crate::segments::SegmentCache;
use crate::{FileType, Footer, SegmentSpec, VortexOpenOptions};

/// A Vortex file that is backed by an in-memory buffer.
///
/// This type of file reader performs no coalescing or other clever orchestration, simply
/// zero-copy slicing the segments from the buffer.
#[derive(Clone)]
pub struct InMemoryVortexFile {
    buffer: ByteBuffer,
    footer: Footer,
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
    type ScanDriver = Self;

    fn scan_driver(
        read: Self::Read,
        _options: Self::Options,
        footer: Footer,
        _segment_cache: Arc<dyn SegmentCache>,
        _metrics: VortexMetrics,
    ) -> Self::ScanDriver {
        Self {
            buffer: read,
            footer,
        }
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

#[async_trait]
impl AsyncSegmentReader for InMemoryVortexFile {
    async fn get(&self, id: SegmentId) -> VortexResult<ByteBuffer> {
        let segment: &SegmentSpec = self
            .footer
            .segment_map()
            .get(*id as usize)
            .ok_or_else(|| vortex_err!("segment not found"))?;

        let start = usize::try_from(segment.offset).map_err(|_| vortex_err!("offset too large"))?;
        let end = start + segment.length as usize;

        Ok(self.buffer.slice(start..end))
    }
}
