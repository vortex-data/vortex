use vortex_array::ArrayRef;
use vortex_error::VortexResult;

use crate::segments::SegmentWriter;
use crate::Layout;

/// A strategy for writing chunks of an array into a layout.
// [layout writer]
pub trait LayoutWriter: Send {
    fn push_chunk(&mut self, segments: &mut dyn SegmentWriter, chunk: ArrayRef)
        -> VortexResult<()>;

    fn finish(&mut self, segments: &mut dyn SegmentWriter) -> VortexResult<Layout>;
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
        segments: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<Layout> {
        self.push_chunk(segments, chunk)?;
        self.finish(segments)
    }

    /// Push all chunks of the iterator into the layout writer and return the finished
    /// [`Layout`].
    fn push_all<I: IntoIterator<Item = VortexResult<ArrayRef>>>(
        &mut self,
        segments: &mut dyn SegmentWriter,
        iter: I,
    ) -> VortexResult<Layout> {
        for chunk in iter.into_iter() {
            self.push_chunk(segments, chunk?)?
        }
        self.finish(segments)
    }
}

impl<L: LayoutWriter + ?Sized> LayoutWriterExt for L {}
