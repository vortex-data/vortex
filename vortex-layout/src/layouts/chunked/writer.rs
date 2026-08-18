// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriter;
use crate::LayoutWriterContext;
use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SequenceId;

#[derive(Clone)]
pub struct ChunkedLayoutStrategy {
    /// The layout strategy for each chunk.
    pub chunk_strategy: Arc<dyn LayoutStrategy>,
}

impl ChunkedLayoutStrategy {
    pub fn new<S: LayoutStrategy>(chunk_strategy: S) -> Self {
        Self {
            chunk_strategy: Arc::new(chunk_strategy),
        }
    }
}

impl LayoutStrategy for ChunkedLayoutStrategy {
    fn new_writer(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        dtype: DType,
        session: &VortexSession,
    ) -> VortexResult<Box<dyn LayoutWriter>> {
        Ok(Box::new(ChunkedLayoutWriter {
            chunk_strategy: Arc::clone(&self.chunk_strategy),
            ctx,
            segment_sink,
            dtype,
            session: session.clone(),
            child_layouts: Vec::new(),
        }))
    }
}

struct ChunkedLayoutWriter {
    chunk_strategy: Arc<dyn LayoutStrategy>,
    ctx: LayoutWriterContext,
    segment_sink: SegmentSinkRef,
    dtype: DType,
    session: VortexSession,
    child_layouts: Vec<LayoutRef>,
}

#[async_trait]
impl LayoutWriter for ChunkedLayoutWriter {
    async fn write(&mut self, sequence_id: SequenceId, chunk: ArrayRef) -> VortexResult<()> {
        let mut sequence = sequence_id.descend();
        let mut child = self.chunk_strategy.new_writer(
            self.ctx.clone(),
            Arc::clone(&self.segment_sink),
            self.dtype.clone(),
            &self.session,
        )?;
        child.write(sequence.advance(), chunk).await?;
        child.finish(sequence.advance()).await?;
        self.child_layouts.push(child.close().await?);
        Ok(())
    }

    async fn finish(&mut self, _sequence_id: SequenceId) -> VortexResult<()> {
        Ok(())
    }

    async fn close(mut self: Box<Self>) -> VortexResult<LayoutRef> {
        let mut child_layouts = std::mem::take(&mut self.child_layouts);
        if child_layouts.len() == 1 {
            Ok(child_layouts.pop().vortex_expect("must have one child"))
        } else {
            let row_count = child_layouts.iter().map(|layout| layout.row_count()).sum();
            Ok(ChunkedLayout::new(
                row_count,
                self.dtype,
                OwnedLayoutChildren::layout_children(child_layouts),
            )
            .into_layout())
        }
    }
}
