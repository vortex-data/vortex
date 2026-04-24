// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use futures::StreamExt;
use futures::TryStreamExt;
use futures::stream;
use vortex_array::ArrayContext;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_io::session::RuntimeSessionExt;
use vortex_session::VortexSession;

use crate::IntoLayout;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt as _;

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
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        mut eof: SequencePointer,
        session: &VortexSession,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let dtype2 = dtype.clone();
        let chunk_strategy = Arc::clone(&self.chunk_strategy);
        let handle = session.handle();

        // We spawn each child to allow parallelism when processing chunks.
        let stream = stream! {
            let mut stream = stream;
            while let Some(chunk) = stream.next().await {
                let chunk_eof = eof.split_off();

                let chunk_strategy = Arc::clone(&chunk_strategy);
                let ctx = ctx.clone();
                let segment_sink = Arc::clone(&segment_sink);
                let dtype = dtype2.clone();
                let session = session.clone();

                yield handle.spawn_nested(move |handle| async move {
                    let session = session.with_handle(handle);
                    chunk_strategy
                        .write_stream(
                            ctx,
                            segment_sink,
                            SequentialStreamAdapter::new(
                                dtype,
                                stream::iter([chunk]),
                            )
                            .sendable(),
                            chunk_eof,
                            &session,
                        )
                        .await
                })
            }
        };

        // Poll all of our children concurrently to accumulate their layouts.
        let mut child_layouts: Vec<LayoutRef> = stream.buffered(usize::MAX).try_collect().await?;

        if child_layouts.len() == 1 {
            Ok(child_layouts.pop().vortex_expect("must have one child"))
        } else {
            let row_count = child_layouts.iter().map(|layout| layout.row_count()).sum();
            Ok(ChunkedLayout::new(
                row_count,
                dtype,
                OwnedLayoutChildren::layout_children(child_layouts),
            )
            .into_layout())
        }
    }

    fn buffered_bytes(&self) -> u64 {
        self.chunk_strategy.buffered_bytes()
    }
}
