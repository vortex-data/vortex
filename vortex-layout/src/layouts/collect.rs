// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_stream::try_stream;
use async_trait::async_trait;
use futures::StreamExt;
use futures::pin_mut;
use vortex_array::ArrayContext;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStream;
use crate::sequence::SequentialStreamAdapter;

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
        segment_sink: SegmentSinkRef,
        stream: SendableSequentialStream,
        eof: SequencePointer,
        session: &VortexSession,
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

            let collected = ChunkedArray::try_new(chunks, _dtype)?.into_array();
            yield (latest_sequence_id.vortex_expect("must have visited at least one chunk"), collected);
        };

        let adapted = Box::pin(SequentialStreamAdapter::new(dtype, collected_stream));

        self.child
            .write_stream(ctx, segment_sink, adapted, eof, session)
            .await
    }

    fn buffered_bytes(&self) -> u64 {
        todo!()
    }
}
