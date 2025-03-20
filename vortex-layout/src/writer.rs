use vortex_array::ArrayRef;
use vortex_error::VortexResult;

use crate::Layout;
use crate::segments::SegmentWriter;

/// A strategy for writing chunks of an array into a layout.
// [layout writer]
pub trait LayoutWriter: Send {
    fn push_chunk(
        &mut self,
        segment_writer: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()>;

    fn finish(&mut self, segment_writer: &mut dyn SegmentWriter) -> VortexResult<Layout>;
}
// [layout writer]

pub trait LayoutWriterExt: LayoutWriter {
    /// Box the layout writer.
    fn boxed(self) -> Box<dyn LayoutWriter>
    where
        Self: Sized + 'static,
    {
        Box::new(self)
    }

    /// Push a single chunk into the layout writer and return the finished [`Layout`].
    fn push_one(
        &mut self,
        segment_writer: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<Layout> {
        self.push_chunk(segment_writer, chunk)?;
        self.finish(segment_writer)
    }

    /// Push all chunks of the iterator into the layout writer and return the finished
    /// [`Layout`].
    fn push_all<I: IntoIterator<Item = VortexResult<ArrayRef>>>(
        &mut self,
        segment_writer: &mut dyn SegmentWriter,
        iter: I,
    ) -> VortexResult<Layout> {
        for chunk in iter.into_iter() {
            self.push_chunk(segment_writer, chunk?)?
        }
        self.finish(segment_writer)
    }
}

impl<L: LayoutWriter + ?Sized> LayoutWriterExt for L {}
