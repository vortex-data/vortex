// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::ArrayRef;
use vortex_array::IntoArray;
use vortex_array::arrays::ChunkedArray;
use vortex_array::dtype::DType;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::BufferedBytesReservation;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriter;
use crate::LayoutWriterContext;
use crate::segments::SegmentSinkRef;
use crate::sequence::SequenceId;

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

impl LayoutStrategy for CollectStrategy {
    fn new_writer(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        dtype: DType,
        session: &VortexSession,
    ) -> VortexResult<Box<dyn LayoutWriter>> {
        let buffered_bytes = ctx.buffered_bytes_tracker().clone();
        Ok(Box::new(CollectLayoutWriter {
            child: self
                .child
                .new_writer(ctx, segment_sink, dtype.clone(), session)?,
            dtype,
            buffered_bytes,
            chunks: Vec::new(),
        }))
    }
}

struct CollectLayoutWriter {
    child: Box<dyn LayoutWriter>,
    dtype: DType,
    buffered_bytes: crate::BufferedBytesTracker,
    chunks: Vec<(SequenceId, ArrayRef, BufferedBytesReservation)>,
}

#[async_trait]
impl LayoutWriter for CollectLayoutWriter {
    async fn write(&mut self, sequence_id: SequenceId, chunk: ArrayRef) -> VortexResult<()> {
        let reservation = self.buffered_bytes.reserve(chunk.nbytes());
        self.chunks.push((sequence_id, chunk, reservation));
        Ok(())
    }

    async fn finish(&mut self, sequence_id: SequenceId) -> VortexResult<()> {
        if !self.chunks.is_empty() {
            let mut chunks = self.chunks.drain(..).collect::<Vec<_>>();
            let (sequence_id, last, last_reservation) =
                chunks.pop().vortex_expect("chunks checked non-empty");
            let chunks = chunks
                .into_iter()
                .map(|(_, chunk, _reservation)| chunk)
                .chain(std::iter::once(last));
            drop(last_reservation);
            let collected = ChunkedArray::try_new(chunks, self.dtype.clone())?.into_array();
            self.child.write(sequence_id, collected).await?;
        }
        self.child.finish(sequence_id).await
    }

    async fn close(self: Box<Self>) -> VortexResult<LayoutRef> {
        self.child.close().await
    }
}
