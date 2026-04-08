// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering;

use async_stream::try_stream;
use async_trait::async_trait;
use futures::StreamExt as _;
use futures::pin_mut;
use vortex_array::ArrayContext;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;

use crate::LayoutRef;
use crate::LayoutStrategy;
use crate::segments::SegmentSinkRef;
use crate::sequence::SendableSequentialStream;
use crate::sequence::SequencePointer;
use crate::sequence::SequentialStreamAdapter;
use crate::sequence::SequentialStreamExt as _;

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
        stream: SendableSequentialStream,
        eof: SequencePointer,
        handle: Handle,
    ) -> VortexResult<LayoutRef> {
        let dtype = stream.dtype().clone();
        let buffer_size = self.buffer_size;

        let buffered_bytes_counter = Arc::clone(&self.buffered_bytes);
        let buffered_stream = try_stream! {
            let stream = stream.peekable();
            pin_mut!(stream);

            let mut nbytes = 0u64;
            let mut chunks = VecDeque::new();

            while let Some(chunk) = stream.as_mut().next().await {
                let (sequence_id, chunk) = chunk?;
                let chunk_size = chunk.nbytes();
                nbytes += chunk_size;
                buffered_bytes_counter.fetch_add(chunk_size, Ordering::Relaxed);
                chunks.push_back(chunk);

                // If this is the last element, flush everything.
                if stream.as_mut().peek().await.is_none() {
                    let mut sequence_ptr = sequence_id.descend();
                    while let Some(chunk) = chunks.pop_front() {
                        buffered_bytes_counter.fetch_sub(chunk.nbytes(), Ordering::Relaxed);
                        yield (sequence_ptr.advance(), chunk)
                    }
                    break;
                }

                if nbytes < 2 * buffer_size {
                    continue;
                };

                // Wait until we're at 2x the buffer size before flushing 1x the buffer size.
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
