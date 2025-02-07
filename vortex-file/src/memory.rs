use std::sync::Arc;

use async_trait::async_trait;
use futures::Stream;
use futures_util::future::BoxFuture;
use futures_util::StreamExt;
use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_err, VortexResult};
use vortex_layout::scan::ScanDriver;
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};

use crate::segments::SegmentCache;
use crate::{FileLayout, FileType, Segment};

/// A Vortex file that is backed by an in-memory buffer.
///
/// This type of file reader performs no coalescing or other clever orchestration, simply
/// zero-copy slicing the segments from the buffer.
#[derive(Clone)]
pub struct InMemoryVortexFile {
    buffer: ByteBuffer,
    file_layout: FileLayout,
}

impl FileType for InMemoryVortexFile {
    type Options = ();
    type Read = ByteBuffer;
    type ScanDriver = Self;

    fn scan_driver(
        read: Self::Read,
        _options: Self::Options,
        file_layout: FileLayout,
        _segment_cache: Arc<dyn SegmentCache>,
    ) -> Self::ScanDriver {
        Self {
            buffer: read,
            file_layout,
        }
    }
}

impl ScanDriver for InMemoryVortexFile {
    type Options = ();

    fn segment_reader(&self) -> Arc<dyn AsyncSegmentReader> {
        Arc::new(self.clone())
    }

    fn drive_stream(
        self,
        stream: impl Stream<Item = BoxFuture<'static, VortexResult<()>>> + Send + 'static,
    ) -> impl Stream<Item = VortexResult<()>> + 'static {
        stream.then(|r| r)
    }
}

#[async_trait]
impl AsyncSegmentReader for InMemoryVortexFile {
    async fn get(&self, id: SegmentId) -> VortexResult<ByteBuffer> {
        let segment: &Segment = self
            .file_layout
            .segment_map()
            .get(*id as usize)
            .ok_or_else(|| vortex_err!("segment not found"))?;

        let start = usize::try_from(segment.offset).map_err(|_| vortex_err!("offset too large"))?;
        let end = start + segment.length as usize;

        Ok(self.buffer.slice(start..end))
    }
}
