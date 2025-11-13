// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;
use std::mem;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_stream::try_stream;
use async_trait::async_trait;
use futures::StreamExt as _;
use vortex_array::{ArrayContext, ArrayRef};
use vortex_error::{VortexExpect, VortexResult};
use vortex_io::runtime::Handle;

use crate::segments::SegmentSinkRef;
use crate::sequence::{
    SendableSequentialStream, SequenceId, SequencePointer, SequentialStreamAdapter,
    SequentialStreamExt as _,
};
use crate::{LayoutRef, LayoutStrategy, Writer};

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

pub struct BufferedWriter {
    next: Box<dyn Writer>,
    options: BufferOptions,
    buffered_chunks: Vec<ArrayRef>,
    buffered_nbytes: u64,
    eof: Option<SequencePointer>,
}

pub struct BufferOptions {
    /// Number of bytes to buffer before flushing
    pub buffer_bytes: u64,
}

impl Writer for BufferedWriter {
    fn init(&mut self, eof: SequencePointer) {
        // Initialize the children with EOF information.
        let (eof, next_eof) = eof.split();
        self.next.init(next_eof);
        self.eof = Some(eof);
    }

    fn push_chunk(&mut self, chunk: ArrayRef, id: SequenceId) -> VortexResult<()> {
        let chunk_bytes = chunk.nbytes();

        if self.buffered_nbytes + chunk_bytes > self.options.buffer_bytes {
            // Flush the buffered chunks. We assign new sequence IDs as a group so they get
            // written together.
            let buffer = mem::take(&mut self.buffered_chunks);
            let mut sequence_ptr = id.descend();

            // NOTE(aduffy): this forces us to call await to flush every chunk.
            // This might not be so bad actually, since we have some buffer nodes in the mix and
            // internally it will hit the buffer before it does anything that is too slow.

            // We buffer all of these and push them together at once.
            for chunk in buffer {
                self.next.push_chunk(chunk, sequence_ptr.advance())?;
            }
        }

        // Buffer the next chunk
        self.buffered_chunks.push(chunk);
        self.buffered_nbytes += chunk_bytes;

        Ok(())
    }

    fn finish(&mut self) -> VortexResult<LayoutRef> {
        // Send any unwritten buffered data to the child
        let mut eof = self.eof.take().vortex_expect("eof saved in initialization");

        let buffer = mem::take(&mut self.buffered_chunks);
        for chunk in buffer {
            self.next.push_chunk(chunk, eof.advance())?;
        }

        drop(eof);

        // Then, we finish the child and return its layout.
        self.next.finish()
    }
}
