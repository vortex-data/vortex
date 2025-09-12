// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::FutureExt;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexExpect, VortexResult, vortex_bail};
use vortex_layout::segments::{SegmentFuture, SegmentId, SegmentSource};

use crate::footer::DeserializeStep;
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
    pub fn open<B: Into<ByteBuffer>>(self, buffer: B) -> VortexResult<VortexFile> {
        let buffer = buffer.into();

        let mut deserializer = Footer::deserializer(buffer.clone())
            .with_size(buffer.len() as u64)
            .with_some_dtype(self.dtype.clone())
            .with_array_registry(self.registry.clone())
            .with_layout_registry(self.layout_registry.clone());

        let footer = match deserializer.deserialize()? {
            DeserializeStep::NeedMoreData { .. } => unreachable!("all data provided up front"),
            DeserializeStep::NeedFileSize => unreachable!("size passed above"),
            DeserializeStep::Done(footer) => footer,
        };

        let segment_source = Arc::new(InMemorySegmentReader {
            buffer,
            footer: footer.clone(),
        });

        Ok(VortexFile {
            footer,
            segment_source,
            metrics: self.metrics,
        })
    }
}

#[derive(Clone)]
struct InMemorySegmentReader {
    buffer: ByteBuffer,
    footer: Footer,
}

impl SegmentSource for InMemorySegmentReader {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let Some(spec) = self.footer.segment_map().get(*id as usize) else {
            return async move { vortex_bail!("segment not found {id}") }.boxed();
        };

        let start = usize::try_from(spec.offset).vortex_expect("segment offset larger than usize");
        let end = start + spec.length as usize;
        let buffer = self.buffer.slice(start..end);

        async move { Ok(buffer) }.boxed()
    }
}
