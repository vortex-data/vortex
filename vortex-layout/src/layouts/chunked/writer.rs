use std::sync::Arc;

use vortex_array::arcref::ArcRef;
use vortex_array::{ArrayContext, ArrayRef};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};

use crate::data::Layout;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::segments::SegmentWriter;
use crate::strategy::LayoutStrategy;
use crate::writer::LayoutWriter;
use crate::{LayoutVTableRef, LayoutWriterExt};

#[derive(Clone)]
pub struct ChunkedLayoutStrategy {
    /// The layout strategy for each chunk.
    pub chunk_strategy: ArcRef<dyn LayoutStrategy>,
}

impl Default for ChunkedLayoutStrategy {
    fn default() -> Self {
        Self {
            chunk_strategy: ArcRef::new_arc(Arc::new(FlatLayoutStrategy::default())),
        }
    }
}

impl LayoutStrategy for ChunkedLayoutStrategy {
    fn new_writer(&self, ctx: &ArrayContext, dtype: &DType) -> VortexResult<Box<dyn LayoutWriter>> {
        Ok(ChunkedLayoutWriter::new(ctx.clone(), dtype.clone(), self.clone()).boxed())
    }
}

/// A basic implementation of a chunked layout writer that writes each batch into its own chunk.
///
/// TODO(ngates): introduce more sophisticated layout writers with different chunking strategies.
pub struct ChunkedLayoutWriter {
    ctx: ArrayContext,
    options: ChunkedLayoutStrategy,
    chunks: Vec<Box<dyn LayoutWriter>>,
    dtype: DType,
    row_count: u64,
}

impl ChunkedLayoutWriter {
    pub fn new(ctx: ArrayContext, dtype: DType, options: ChunkedLayoutStrategy) -> Self {
        Self {
            ctx,
            options,
            chunks: vec![],
            dtype,
            row_count: 0,
        }
    }
}

impl LayoutWriter for ChunkedLayoutWriter {
    fn push_chunk(
        &mut self,
        segment_writer: &mut dyn SegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        self.row_count += chunk.len() as u64;

        // We write each chunk, but don't call finish quite yet to ensure that chunks have an
        // opportunity to write messages at the end of the file.
        let mut chunk_writer = self
            .options
            .chunk_strategy
            .new_writer(&self.ctx, chunk.dtype())?;
        chunk_writer.push_chunk(segment_writer, chunk)?;
        chunk_writer.flush(segment_writer)?;
        self.chunks.push(chunk_writer);

        Ok(())
    }

    fn flush(&mut self, _segment_writer: &mut dyn SegmentWriter) -> VortexResult<()> {
        // We flush each chunk as we write it, so there's nothing to do here.
        Ok(())
    }

    fn finish(&mut self, segment_writer: &mut dyn SegmentWriter) -> VortexResult<Layout> {
        // Call finish on each chunk's writer
        let mut children = vec![];
        for writer in self.chunks.iter_mut() {
            // FIXME(ngates): we should try calling finish after each chunk.
            children.push(writer.finish(segment_writer)?);
        }

        // If there's only one child, there's no point even writing a stats table since
        // there's no pruning for us to do.
        if children.len() == 1 {
            return Ok(children.pop().vortex_expect("child layout"));
        }

        Ok(Layout::new_owned(
            "chunked".into(),
            LayoutVTableRef::new_ref(&ChunkedLayout),
            self.dtype.clone(),
            self.row_count,
            vec![],
            children,
            None,
        ))
    }
}
