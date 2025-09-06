// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use futures::stream::once;
use futures::StreamExt;
use vortex_array::ArrayContext;
use vortex_error::{VortexExpect, VortexResult};
use vortex_io::runtime::Handle;

use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::segments::SegmentSink;
use crate::sequence::{
    SendableSequentialStream, SequencePointer, SequentialStreamAdapter, SequentialStreamExt as _,
};
use crate::{IntoLayout, LayoutRef, LayoutStrategy};

#[derive(Clone)]
pub struct ChunkedLayoutStrategy<S> {
    /// The layout strategy for each chunk.
    pub chunk_strategy: S,
}

impl<S> ChunkedLayoutStrategy<S>
where
    S: LayoutStrategy,
{
    pub fn new(chunk_strategy: S) -> Self {
        Self { chunk_strategy }
    }
}

#[async_trait]
impl<S> LayoutStrategy for ChunkedLayoutStrategy<S>
where
    S: LayoutStrategy,
{
    async fn write_stream<'rt>(
        &self,
        ctx: &ArrayContext,
        segment_sink: &dyn SegmentSink,
        mut stream: SendableSequentialStream,
        mut eof: SequencePointer,
        handle: Handle<'rt>,
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
                    segment_sink,
                    SequentialStreamAdapter::new(
                        dtype.clone(),
                        once(async { Ok((sequence_id, chunk)) }),
                    )
                    .sendable(),
                    eof.advance().descend(),
                    handle.clone(),
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
