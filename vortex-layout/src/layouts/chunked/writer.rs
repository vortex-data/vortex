use std::pin::Pin;
use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use futures::stream::once;
use arcref::ArcRef;
use vortex_array::{ArrayContext, ArrayRef};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};

use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::flat::writer::{FlatLayoutStrategy, NewFlatLayoutStrategy};
use crate::segments::{ConcurrentSegmentWriter, NewSegmentWriter};
use crate::strategy::LayoutStrategy;
use crate::writer::LayoutWriter;
use crate::{
    IntoLayout, LayoutRef, LayoutWriterExt, NewLayoutStrategy, NewLayoutWriter, SequentialArrayStream,
};

pub struct NewChunkedLayoutStrategy {
    /// The layout strategy for each chunk.
    pub chunk_strategy: ArcRef<dyn NewLayoutStrategy>,
}

impl Default for NewChunkedLayoutStrategy {
    fn default() -> Self {
        Self {
            chunk_strategy: ArcRef::new_arc(Arc::new(NewFlatLayoutStrategy::default())),
        }
    }
}

impl NewLayoutStrategy for NewChunkedLayoutStrategy {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        dtype: &DType,
        segment_writer: Arc<dyn NewSegmentWriter>,
        mut stream: SequentialArrayStream,
    ) -> Pin<Box<dyn NewLayoutWriter>> {
        let chunk_strategy = self.chunk_strategy.clone();
        let ctx = ctx.clone();
        let dtype = dtype.clone();
        Box::pin(async move {
            let mut child_layouts = Vec::new();
            let mut row_count = 0;
            while let Some(chunk) = stream.next().await {
                let (sequence_id, chunk) = chunk?;
                row_count += chunk.len() as u64;
                let layout = chunk_strategy
                    .write_stream(
                        &ctx,
                        &dtype,
                        segment_writer.clone(),
                        Box::pin(once(async { Ok((sequence_id, chunk)) })),
                    )
                    .await?;
                child_layouts.push(layout);
            }

            if child_layouts.len() == 1 {
                Ok(child_layouts.pop().vortex_expect("must have one child"))
            } else {
                Ok(chunked_layout(dtype, row_count, child_layouts))
            }
        })
    }
}

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

#[async_trait]
impl LayoutWriter for ChunkedLayoutWriter {
    async fn push_chunk(
        &mut self,
        segment_writer: &mut dyn ConcurrentSegmentWriter,
        chunk: ArrayRef,
    ) -> VortexResult<()> {
        assert_eq!(
            chunk.dtype(),
            &self.dtype,
            "Can't push chunks of the wrong dtype into a LayoutWriter. Pushed {} but expected {}.",
            chunk.dtype(),
            self.dtype
        );

        self.row_count += chunk.len() as u64;

        // We write each chunk, but don't call finish quite yet to ensure that chunks have an
        // opportunity to write messages at the end of the file.
        let mut chunk_writer = self
            .options
            .chunk_strategy
            .new_writer(&self.ctx, chunk.dtype())?;
        chunk_writer.push_chunk(segment_writer, chunk).await?;
        chunk_writer.flush(segment_writer).await?;
        self.chunks.push(chunk_writer);

        Ok(())
    }

    async fn flush(
        &mut self,
        _segment_writer: &mut dyn ConcurrentSegmentWriter,
    ) -> VortexResult<()> {
        // We flush each chunk as we write it, so there's nothing to do here.
        Ok(())
    }

    async fn finish(
        &mut self,
        segment_writer: &mut dyn ConcurrentSegmentWriter,
    ) -> VortexResult<LayoutRef> {
        // Call finish on each chunk's writer
        let mut children = vec![];
        for writer in self.chunks.iter_mut() {
            // FIXME(ngates): we should try calling finish after each chunk.
            children.push(writer.finish(segment_writer).await?);
        }

        // If there's only one child, there's no point even writing a stats table since
        // there's no pruning for us to do.
        if children.len() == 1 {
            return Ok(children.pop().vortex_expect("child layout"));
        }

        Ok(ChunkedLayout::new(
            self.row_count,
            self.dtype.clone(),
            OwnedLayoutChildren::layout_children(children),
        )
        .into_layout())
    }
}
