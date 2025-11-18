// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_stream::try_stream;
use async_trait::async_trait;
use futures::StreamExt as _;
use vortex_array::ArrayContext;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;

use crate::segments::SegmentSinkRef;
use crate::sequence::{
    SendableSequentialStream, SequencePointer, SequentialStreamAdapter, SequentialStreamExt as _,
};
use crate::{LayoutRef, LayoutStrategy};

#[derive(Clone)]
pub struct BufferedStrategy {
    child: Arc<dyn LayoutStrategy>,
    buffer_size: u64,
    buffered_bytes: Arc<AtomicU64>,
}

impl BufferedStrategy {
    pub fn new<S: LayoutStrategy>(child: S, buffer_size: u64) -> Self {
        Self {
            child: Arc::new(child),
            buffer_size,
            buffered_bytes: Arc::new(AtomicU64::new(0)),
        }
    }
}

#[async_trait]
impl LayoutStrategy for BufferedStrategy {
    async fn write_stream(
        &self,
        ctx: ArrayContext,
        segment_sink: SegmentSinkRef,
        mut stream: SendableSequentialStream,
        mut eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let buffer_size = self.buffer_size;

        // We have no choice but to put our final buffers here!
        // We cannot hold on to sequence ids across iterations of the stream, otherwise we can
        // cause deadlocks with other columns that are waiting for us to flush.
        let mut final_flush = eof.split_off();

        let buffered_bytes_counter = self.buffered_bytes.clone();
        let buffered_stream = try_stream! {
            let mut nbytes = 0u64;
            let mut chunks = VecDeque::new();

            while let Some(chunk) = stream.as_mut().next().await {
                let (sequence_id, chunk) = chunk?;
                let chunk_size = chunk.nbytes();
                nbytes += chunk_size;
                buffered_bytes_counter.fetch_add(chunk_size, Ordering::Relaxed);
                chunks.push_back(chunk);

                if nbytes < 2 * buffer_size {
                    continue;
                };

                // Wait until we're at 2x the buffer size before flushing 1x the buffer size
                // This avoids small tail stragglers being flushed at the end of the file.
                let mut sequence_ptr = sequence_id.descend();
                while nbytes > buffer_size {
                    let Some(chunk) = chunks.pop_front() else {
                        break;
                    };
                    let chunk_size = chunk.nbytes();
                    nbytes -= chunk_size;
                    buffered_bytes_counter.fetch_sub(chunk_size, Ordering::Relaxed);
                    yield (sequence_ptr.advance(), chunk)
                }
            }

            // Now the input stream has ended, flush everything
            while let Some(chunk) = chunks.pop_front() {
                let chunk_size = chunk.nbytes();
                buffered_bytes_counter.fetch_sub(chunk_size, Ordering::Relaxed);
                yield (final_flush.advance(), chunk)
            }
        };

        self.child
            .write_stream(
                ctx,
                segment_sink,
                SequentialStreamAdapter::new(dtype, buffered_stream).sendable(),
                eof,
                handle,
            )
            .await
    }

    fn buffered_bytes(&self) -> u64 {
        self.buffered_bytes.load(Ordering::Relaxed) + self.child.buffered_bytes()
    }
}
