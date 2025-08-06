// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use futures::stream::once;
use futures::StreamExt;
use vortex_array::ArrayContext;
use vortex_error::{VortexExpect, VortexResult};

use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::segments::SequenceWriter;
use crate::{
    IntoLayout, LayoutRef, LayoutStrategy, SendableSequentialStream, SequentialStreamAdapter,
    SequentialStreamExt as _,
};

#[derive(Clone)]
pub struct ChunkedStrategy<S> {
    /// The layout strategy for each chunk.
    pub chunk_strategy: S,
}

impl<S> ChunkedStrategy<S>
where
    S: LayoutStrategy,
{
    pub fn new(child: S) -> Self {
        Self {
            chunk_strategy: child,
        }
    }
}

#[async_trait]
impl<S> LayoutStrategy for ChunkedStrategy<S>
where
    S: LayoutStrategy,
{
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        mut stream: SendableSequentialStream,
    ) -> VortexResult<LayoutRef> {
        let ctx = ctx.clone();
        let mut child_layouts = Vec::new();
        let mut row_count = 0;
        let dtype = stream.dtype().clone();
        while let Some(chunk) = stream.next().await {
            let (sequence_id, chunk) = chunk?;
            row_count += chunk.len() as u64;
            let layout = self
                .chunk_strategy
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
    }
}
