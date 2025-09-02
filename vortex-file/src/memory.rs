// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::FutureExt;
use vortex_buffer::ByteBuffer;
use vortex_error::{vortex_bail, vortex_err, VortexExpect, VortexResult};
use vortex_io::runtime::source::IoSource;
use vortex_io::runtime::Handle;
use vortex_layout::segments::{SegmentFuture, SegmentId, SegmentSource};

use crate::{FileType, Footer, VortexFile, VortexOpenOptions};

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
    pub fn open<B: Into<ByteBuffer>>(self, buffer: B, handle: Handle) -> VortexResult<VortexFile> {
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

        let name = buffer.name();

        Ok(VortexFile {
            footer,
            file: handle.open(buffer),
            metrics: self.metrics,
            handle,
            _span: tracing::span!(tracing::Level::INFO, "VortexFile", name = ?name),
        })
    }
}

#[derive(Clone)]
struct InMemorySegmentReader {
    buffer: ByteBuffer,
    footer: Footer,
}

impl SegmentSource<'static> for InMemorySegmentReader {
    fn request(&self, id: SegmentId) -> SegmentFuture<'static> {
        let Some(spec) = self.footer.segment_map().get(*id as usize) else {
            return async move { vortex_bail!("segment not found {id}") }.boxed();
        };

        let start = usize::try_from(spec.offset).vortex_expect("segment offset larger than usize");
        let end = start + spec.length as usize;
        let buffer = self.buffer.slice(start..end);

        async move { Ok(buffer) }.boxed()
    }
}
