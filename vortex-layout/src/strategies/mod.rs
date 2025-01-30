//! This is a collection of built-in layout strategies designed to be used in conjunction with one
//! another to develop an overall strategy.
//!
//! Each [`LayoutWriter`] is passed horizontal chunks of a Vortex array one-by-one, and is
//! eventually asked to return a [`LayoutData`]. The writers can buffer, re-chunk, flush, or
//! otherwise manipulate the chunks of data enabling experimentation with different strategies
//! all while remaining independent of the read code.

mod strategy;

pub use strategy::*;
use vortex_array::Array;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::layouts::flat::writer::FlatLayoutWriter;
use crate::layouts::flat::FlatLayout;
use crate::segments::SegmentWriter;
use crate::LayoutData;

/// A strategy for writing chunks of an array into a layout.
/// FIXME(ngates): move this into writer.rs
// [layout writer]
pub trait LayoutWriter: Send {
    fn push_chunk(&mut self, segments: &mut dyn SegmentWriter, chunk: Array) -> VortexResult<()>;

    fn finish(&mut self, segments: &mut dyn SegmentWriter) -> VortexResult<LayoutData>;
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

    /// Push a single chunk into the layout writer and return the finished [`LayoutData`].
    fn push_one(
        &mut self,
        segments: &mut dyn SegmentWriter,
        chunk: Array,
    ) -> VortexResult<LayoutData> {
        self.push_chunk(segments, chunk)?;
        self.finish(segments)
    }

    /// Push all chunks of the iterator into the layout writer and return the finished
    /// [`LayoutData`].
    fn push_all<I: IntoIterator<Item = VortexResult<Array>>>(
        &mut self,
        segments: &mut dyn SegmentWriter,
        iter: I,
    ) -> VortexResult<LayoutData> {
        for chunk in iter.into_iter() {
            self.push_chunk(segments, chunk?)?
        }
        self.finish(segments)
    }
}

impl<L: LayoutWriter> LayoutWriterExt for L {}

/// A trait for creating new layout writers given a DType.
pub trait LayoutStrategy: Send + Sync {
    fn new_writer(&self, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>>;
}

/// Implement the [`LayoutStrategy`] trait for the [`FlatLayout`] for easy use.
impl LayoutStrategy for FlatLayout {
    fn new_writer(&self, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        Ok(FlatLayoutWriter::new(dtype.clone(), Default::default()).boxed())
    }
}
