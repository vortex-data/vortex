// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use arcref::ArcRef;
use futures::StreamExt;
use futures::stream::once;
use vortex_array::ArrayContext;
use vortex_error::VortexExpect;

use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::layouts::flat::writer::FlatLayoutStrategy;
use crate::segments::SequenceWriter;
use crate::{
    IntoLayout, LayoutStrategy, SendableLayoutWriter, SendableSequentialStream,
    SequentialStreamAdapter, SequentialStreamExt as _,
};

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
        sequence_writer: SequenceWriter,
        mut stream: SendableSequentialStream,
    ) -> SendableLayoutWriter {
        let chunk_strategy = self.chunk_strategy.clone();
        let ctx = ctx.clone();
        Box::pin(async move {
            let mut child_layouts = Vec::new();
            let mut row_count = 0;
            let dtype = stream.dtype().clone();
            while let Some(chunk) = stream.next().await {
                let (sequence_id, chunk) = chunk?;
                row_count += chunk.len() as u64;
                let layout = chunk_strategy
                    .write_stream(
                        &ctx,
                        sequence_writer.clone(),
                        SequentialStreamAdapter::new(
                            dtype.clone(),
                            once(async { Ok((sequence_id, chunk)) }),
                        )
                        .sendable(),
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
