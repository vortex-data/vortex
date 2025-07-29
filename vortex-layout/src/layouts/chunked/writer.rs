// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::StreamExt;
use vortex_array::ArrayContext;
use vortex_error::{VortexExpect, VortexResult};

use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::segments::SegmentSink;
use crate::sequence::SequencePointer;
use crate::{
    IntoLayout, LayoutRef, LayoutStrategy, SendableSequentialStream, SequentialArrayStreamExt,
    TaskExecutor,
};

pub struct ChunkedLayoutStrategy {
    /// The layout strategy for each chunk.
    pub chunk_strategy: Arc<dyn LayoutStrategy>,
}

impl Default for ChunkedLayoutStrategy {
    fn default() -> Self {
        Self {
            chunk_strategy: Arc::new(FlatLayoutStrategy::default()),
        }
    }
}

#[async_trait(?Send)]
impl LayoutStrategy for ChunkedLayoutStrategy {
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        segment_sink: &dyn SegmentSink,
        executor: &Arc<dyn TaskExecutor>,
        mut stream: SendableSequentialStream,
        mut end_of_file: SequencePointer,
    ) -> VortexResult<LayoutRef> {
        let chunk_strategy = self.chunk_strategy.clone();
        let ctx = ctx.clone();
        let mut child_layouts = Vec::new();
        let mut row_count = 0;

        let dtype = stream.dtype().clone();
        while let Some(chunk) = stream.next().await {
            let (seq_id, chunk) = chunk?;
            row_count += chunk.len() as u64;

            let layout = chunk_strategy
                .write_stream(
                    &ctx,
                    segment_sink,
                    executor,
                    chunk.to_array_stream().sequenced(seq_id.descend()),
                    end_of_file.advance().descend(),
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
    }
}
