// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use futures::{StreamExt, pin_mut};
use vortex_array::ArrayContext;
use vortex_array::arrays::ChunkedArray;
use vortex_error::{VortexExpect, VortexResult};
use vortex_io::runtime::Handle;
use vortex_session::VortexSession;

use crate::segments::SegmentSinkRef;
use crate::sequence::{
    SendableSequentialStream, SequencePointer, SequentialStream, SequentialStreamAdapter,
};
use crate::{LayoutRef, LayoutStrategy};

/// A strategy that collects all chunks and turns them into a single array chunk to pass into
/// a child strategy.
pub struct CollectStrategy {
    child: Arc<dyn LayoutStrategy>,
}

impl CollectStrategy {
    pub fn new<S: LayoutStrategy>(child: S) -> CollectStrategy {
        CollectStrategy {
            child: Arc::new(child),
        }
    }
}

#[async_trait]
impl LayoutStrategy for CollectStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        session: &VortexSession,
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        // Read the whole stream, then write one Chunked stream to the inner thing
        let dtype = stream.dtype().clone();

        let _dtype = dtype.clone();
        let collected_stream = try_stream! {
            pin_mut!(stream);

            let mut chunks = Vec::new();
            let mut latest_sequence_id = None;
            while let Some(chunk) = stream.next().await {
                let (sequence_id, chunk) = chunk?;
                latest_sequence_id = Some(sequence_id);
                chunks.push(chunk);
            }

            let collected = ChunkedArray::try_new(chunks, _dtype)?.to_array();
            yield (latest_sequence_id.vortex_expect("must have visited at least one chunk"), collected);
        };

        let adapted = Box::pin(SequentialStreamAdapter::new(dtype, collected_stream));

        self.child
            .write_stream(ctx, session, segment_sink, adapted, eof, handle)
            .await
    }

    fn buffered_bytes(&self) -> u64 {
        todo!()
    }
}
