// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem;
use std::sync::Arc;

use async_stream::stream;
use async_trait::async_trait;
use futures::{StreamExt, TryStreamExt, stream};
use vortex_array::{ArrayContext, ArrayRef};
use vortex_dtype::DType;
use vortex_error::{VortexExpect, VortexResult};
use vortex_io::runtime::Handle;

use crate::children::OwnedLayoutChildren;
use crate::layouts::chunked::ChunkedLayout;
use crate::segments::SegmentSinkRef;
use crate::sequence::{
    SendableSequentialStream, SequenceId, SequencePointer, SequentialStreamAdapter,
    SequentialStreamExt as _,
};
use crate::{IntoLayout, LayoutRef, LayoutStrategy, Writer};

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
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let dtype2 = dtype.clone();
        let chunk_strategy = self.chunk_strategy.clone();

        // We spawn each child to allow parallelism when processing chunks.
        let stream = stream! {
            let mut stream = stream;
            while let Some(chunk) = stream.next().await {
                let chunk_eof = eof.split_off();

                let chunk_strategy = chunk_strategy.clone();
                let ctx = ctx.clone();
                let segment_sink = segment_sink.clone();
                let dtype = dtype2.clone();

                yield handle.spawn_nested(move |handle| async move {
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
                            handle,
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

pub struct ChunkedWriter {
    eof: Option<SequencePointer>,
    dtype: DType,
    make_writer: Box<dyn Fn() -> Box<dyn Writer> + Send + Sync + 'static>,
    layouts: Vec<LayoutRef>,
}

impl ChunkedWriter {
    pub fn new(
        dtype: DType,
        make_writer: Box<dyn Fn() -> Box<dyn Writer> + Send + Sync + 'static>,
    ) -> Self {
        Self {
            make_writer,
            dtype,
            eof: None,
            layouts: Vec::new(),
        }
    }
}

#[async_trait]
impl Writer for ChunkedWriter {
    fn init(&mut self, eof: SequencePointer) {
        self.eof = Some(eof);
    }

    async fn push_chunk(&mut self, chunk: ArrayRef, id: SequenceId) -> VortexResult<()> {
        let mut eof = self.eof.take().vortex_expect("eof must be present");
        let chunk_eof = eof.split_off();
        self.eof = Some(eof);

        let mut next = (self.make_writer)();
        next.init(chunk_eof);
        next.push_chunk(chunk, id).await?;

        self.layouts.push(next.finish().await?);

        Ok(())
    }

    async fn finish(&mut self) -> VortexResult<LayoutRef> {
        let layouts = mem::take(&mut self.layouts);
        let row_count = layouts.iter().map(|layout| layout.row_count()).sum::<u64>();

        Ok(ChunkedLayout::new(
            row_count,
            self.dtype.clone(),
            Arc::new(OwnedLayoutChildren::layout_children(layouts)),
        )
        .into_layout())
    }
}
