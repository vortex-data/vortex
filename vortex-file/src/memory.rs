use std::sync::Arc;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_layout::segments::{SegmentId, SegmentSource};
use vortex_metrics::VortexMetrics;

use crate::{FileType, Footer, SegmentSourceFactory, SegmentSpec, VortexFile, VortexOpenOptions};

/// A Vortex file that is backed by an in-memory buffer.
///
/// This type of file reader performs no coalescing or other clever orchestration, simply
/// zero-copy slicing the segments from the buffer.
pub struct InMemoryFileType;

impl FileType for InMemoryFileType {
    type Options = ();
}

impl VortexOpenOptions<InMemoryFileType> {
    /// Create open options for an in-memory Vortex file.
    pub fn in_memory() -> Self {
        Self::new(())
    }

    /// Open an in-memory file contained in the provided buffer.
    pub async fn open<B: Into<ByteBuffer>>(self, buffer: B) -> VortexResult<VortexFile> {
        let buffer = buffer.into();

        let postscript = self.parse_postscript(&buffer)?;

        // If we haven't been provided a DType, we must read one from the file.
        let dtype = self.dtype
            .clone()
            .map(Ok)
            .unwrap_or_else(|| {
                let dtype_segment = postscript
                    .dtype
                    .ok_or_else(|| vortex_err!("Vortex file doesn't embed a DType and one has not been provided to VortexOpenOptions"))?;
                self.parse_dtype(0, &buffer, &dtype_segment)
            })?;

        let file_stats = postscript
            .statistics
            .map(|segment| self.parse_file_statistics(0, &buffer, &segment))
            .transpose()?;

        let footer = self.parse_footer(
            0,
            &buffer,
            &postscript.footer,
            &postscript.layout,
            dtype,
            file_stats,
        )?;

        let segment_source_factory = Arc::new(InMemorySegmentReader {
            buffer,
            footer: footer.clone(),
        });

        Ok(VortexFile {
            footer,
            segment_source_factory,
            metrics: self.metrics,
        })
    }
}

#[derive(Clone)]
struct InMemorySegmentReader {
    buffer: ByteBuffer,
    footer: Footer,
}

impl SegmentSourceFactory for InMemorySegmentReader {
    fn segment_source(&self, _metrics: VortexMetrics) -> Arc<dyn SegmentSource> {
        Arc::new(self.clone())
    }
}

impl SegmentSource for InMemorySegmentReader {
    fn request(
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
                usize::try_from(segment.offset).vortex_expect("segment offset larger than usize");
            let end = start + segment.length as usize;

            Ok(buffer.slice(start..end))
        }
        .boxed()
    }
}
