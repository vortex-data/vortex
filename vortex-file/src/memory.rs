use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::ContextRef;
use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_err, VortexResult};
use vortex_layout::scan::ScanDriver;
use vortex_layout::segments::{AsyncSegmentReader, SegmentId};

use crate::segments::SegmentCache;
use crate::{FileLayout, Segment, VortexFileOpener};

#[derive(Clone)]
pub struct InMemoryVortexFile {
    buffer: ByteBuffer,
    file_layout: FileLayout,
    ctx: ContextRef,
}

impl VortexFileOpener for InMemoryVortexFile {
    type Options = ();
    type Read = ByteBuffer;
    type ScanDriver = Self;

    fn open(
        ctx: ContextRef,
        file_layout: FileLayout,
        _segment_cache: Arc<dyn SegmentCache>,
        _options: Self::Options,
        read: Self::Read,
    ) -> VortexResult<Self> {
        Ok(Self {
            buffer: read,
            file_layout,
            ctx,
        })
    }

    fn scan_driver(&self) -> Self::ScanDriver {
        self.clone()
    }
}

impl ScanDriver for InMemoryVortexFile {
    type Options = ();

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
