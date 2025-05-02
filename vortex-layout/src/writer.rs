use async_trait::async_trait;
use vortex_array::ArrayRef;
use vortex_error::VortexResult;

use crate::LayoutRef;
use crate::segments::SegmentWriter;

/// A strategy for writing chunks of an array into a layout.
// [layout writer]
#[async_trait]
pub trait LayoutWriter: Send {
    /// Push a chunk into the layout writer.
    async fn push_chunk(
        &mut self,
        segment_writer: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()>;

    /// Flush any buffered chunks.
    async fn flush(&mut self, segment_writer: &mut dyn SegmentWriter) -> VortexResult<()>;

    /// Write any final data (e.g. stats) and return the finished [`LayoutRef`].
    async fn finish(&mut self, segment_writer: &mut dyn SegmentWriter) -> VortexResult<LayoutRef>;
}
// [layout writer]

#[async_trait]
pub trait LayoutWriterExt: LayoutWriter {
    /// Box the layout writer.
    fn boxed(self) -> Box<dyn LayoutWriter>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }

    /// Push a single chunk into the layout writer and return the finished [`LayoutRef`].
    async fn push_one(
        &mut self,
        segment_writer: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<LayoutRef> {
        self.push_chunk(segment_writer, chunk)?;
        self.flush(segment_writer)?;
        self.finish(segment_writer)
    }

    /// Push all chunks of the iterator into the layout writer and return the finished
    /// [`LayoutRef`].
    async fn push_all<I: IntoIterator<Item = VortexResult<ArrayRef>>>(
        &mut self,
        segment_writer: &mut dyn SegmentWriter,
        iter: I,
    ) -> VortexResult<LayoutRef> {
        for chunk in iter.into_iter() {
            self.push_chunk(segment_writer, chunk?).await?
        }
        self.flush(segment_writer).await?;
        self.finish(segment_writer).await
    }
}

impl<L: LayoutWriter + ?Sized> LayoutWriterExt for L {}
