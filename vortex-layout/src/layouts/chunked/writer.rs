use std::pin::Pin;
use std::sync::Arc;

use arcref::ArcRef;
use futures::StreamExt;
use futures::stream::once;
use vortex_array::ArrayContext;
use vortex_dtype::DType;
use vortex_error::VortexExpect;

use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::segments::SegmentWriter;
use crate::{IntoLayout, LayoutStrategy, LayoutWriter, SequentialArrayStream};

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
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        dtype: &DType,
        segment_writer: Arc<dyn SegmentWriter>,
        mut stream: SequentialArrayStream,
    ) -> Pin<Box<dyn LayoutWriter>> {
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
                Ok(ChunkedLayout::new(
                    row_count,
                    dtype,
                    OwnedLayoutChildren::layout_children(child_layouts),
                )
                .into_layout())
            }
        })
    }
}
