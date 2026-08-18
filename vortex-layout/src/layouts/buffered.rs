// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;
use vortex_array::ArrayRef;
use vortex_array::dtype::DType;
use vortex_error::VortexResult;
use vortex_session::VortexSession;

use crate::BufferedBytesReservation;
use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::LayoutWriter;
use crate::LayoutWriterContext;
use crate::segments::SegmentSinkRef;
use crate::sequence::SequenceId;

#[derive(Clone)]
pub struct BufferedStrategy {
    child: Arc<dyn LayoutStrategy>,
    buffer_size: u64,
}

impl BufferedStrategy {
    pub fn new<S: LayoutStrategy>(child: S, buffer_size: u64) -> Self {
        Self {
            child: Arc::new(child),
            buffer_size,
        }
    }
}

impl LayoutStrategy for BufferedStrategy {
    fn new_writer(
        &self,
        ctx: LayoutWriterContext,
        segment_sink: SegmentSinkRef,
        dtype: DType,
        session: &VortexSession,
    ) -> VortexResult<Box<dyn LayoutWriter>> {
        let buffered_bytes = ctx.buffered_bytes_tracker().clone();
        Ok(Box::new(BufferedLayoutWriter {
            child: self.child.new_writer(ctx, segment_sink, dtype, session)?,
            buffer_size: self.buffer_size,
            buffered_bytes,
            pending: None,
            chunks: VecDeque::new(),
            nbytes: 0,
        }))
    }
}

struct BufferedLayoutWriter {
    child: Box<dyn LayoutWriter>,
    buffer_size: u64,
    buffered_bytes: crate::BufferedBytesTracker,
    pending: Option<(SequenceId, ArrayRef, BufferedBytesReservation)>,
    chunks: VecDeque<(SequenceId, ArrayRef, BufferedBytesReservation)>,
    nbytes: u64,
}

impl BufferedLayoutWriter {
    async fn process(
        &mut self,
        sequence_id: SequenceId,
        chunk: ArrayRef,
        reservation: BufferedBytesReservation,
        last: bool,
    ) -> VortexResult<()> {
        self.nbytes += reservation.bytes();
        self.chunks.push_back((sequence_id, chunk, reservation));

        if !last && self.nbytes < 2 * self.buffer_size {
            return Ok(());
        }

        while last || self.nbytes > self.buffer_size {
            let Some((sequence_id, chunk, reservation)) = self.chunks.pop_front() else {
                break;
            };
            self.nbytes -= reservation.bytes();
            drop(reservation);
            self.child.write(sequence_id, chunk).await?;
        }
        Ok(())
    }
}

#[async_trait]
impl LayoutWriter for BufferedLayoutWriter {
    async fn write(&mut self, sequence_id: SequenceId, chunk: ArrayRef) -> VortexResult<()> {
        let reservation = self.buffered_bytes.reserve(chunk.nbytes());
        if let Some((pending_id, pending, pending_reservation)) =
            self.pending.replace((sequence_id, chunk, reservation))
        {
            self.process(pending_id, pending, pending_reservation, false)
                .await?;
        }
        Ok(())
    }

    async fn finish(&mut self, sequence_id: SequenceId) -> VortexResult<()> {
        if let Some((sequence_id, chunk, reservation)) = self.pending.take() {
            self.process(sequence_id, chunk, reservation, true).await?;
        }
        self.child.finish(sequence_id).await
    }

    async fn close(self: Box<Self>) -> VortexResult<LayoutRef> {
        self.child.close().await
    }
}
