// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::{ StreamExt, TryStreamExt};
use futures::stream::once;
use vortex_array::{Array, ArrayContext};
use vortex_error::{ VortexExpect, VortexResult};
use vortex_io::runtime::Handle;

use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::segments::SegmentSink;
use crate::sequence::{SendableSequentialStream, SequencePointer,  SequentialStreamAdapter, SequentialStreamExt as _};
use crate::{IntoLayout, LayoutRef, LayoutStrategy};

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

#[async_trait]
impl LayoutStrategy for ChunkedLayoutStrategy {
    async fn write_stream(
        &self,
        ctx: &ArrayContext,
        segment_sink: &Arc<dyn SegmentSink>,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        handle: &Handle,
    ) -> VortexResult<LayoutRef> {
        let mut row_count = 0;
        let dtype = stream.dtype().clone();
        let dtype2 = dtype.clone();

        let mut child_layouts: Vec<_>= stream
            .map(move |chunk| {
                let dtype = dtype.clone();
                let chunk_strategy = self.chunk_strategy.clone();
                let eof = eof.advance().descend();
                async move {
                    let (sequence_id, chunk) = chunk?;
                    // row_count += chunk.len() as u64;
                    chunk_strategy
                        .write_stream(
                            ctx,
                            segment_sink,
                            SequentialStreamAdapter::new(
                                dtype,
                                once(async { Ok((sequence_id, chunk)) }),
                            )
                                .sendable(),
                            eof,
                            handle,
                        ).await
                }
            })
            .buffered(16)
            .try_collect()
            .await?;

        if child_layouts.len() == 1 {
            Ok(child_layouts.pop().vortex_expect("must have one child"))
        } else {
            Ok(ChunkedLayout::new(
                row_count,
                dtype2,
                OwnedLayoutChildren::layout_children(child_layouts),
            )
            .into_layout())
        }
    }
}
