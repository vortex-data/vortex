use std::sync::Arc;

use async_trait::async_trait;
use futures_executor::block_on;
use vortex_array::ContextRef;
use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_err, VortexResult};
use vortex_layout::scan::{Scan, ScanDriver};
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};

use crate::segments::NoOpSegmentCache;
use crate::{FileLayout, Segment, VortexOpenOptions};

impl VortexOpenOptions {
    /// Open an in-memory file contained in the provided buffer.
    pub fn open_memory<B: Into<ByteBuffer>>(self, buffer: B) -> VortexResult<InMemoryVortexFile> {
        let buffer = buffer.into();

        // No point caching anything, and we can block_on since I/O is basically free.
        let file_layout = block_on(self.read_file_layout(&buffer, &NoOpSegmentCache))?;

        Ok(InMemoryVortexFile {
            buffer,
            file_layout,
            ctx: self.ctx,
        })
    }
}

#[derive(Clone)]
pub struct InMemoryVortexFile {
    buffer: ByteBuffer,
    file_layout: FileLayout,
    ctx: ContextRef,
}

impl InMemoryVortexFile {
    pub fn scan(self) -> Scan<InMemoryVortexFile> {
        let layout = self.file_layout.root_layout().clone();
        let ctx = self.ctx.clone();
        Scan::new(self, layout, ctx)
    }
}

impl ScanDriver for InMemoryVortexFile {
    fn segment_reader(&self) -> Arc<dyn AsyncSegmentReader> {
        Arc::new(self.clone())
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
