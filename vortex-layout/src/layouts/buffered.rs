// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;

use arcref::ArcRef;
use async_stream::try_stream;
use futures::{StreamExt as _, pin_mut};
use vortex_array::ArrayContext;

use crate::segments::SequenceWriter;
use crate::{
    LayoutStrategy, SendableLayoutWriter, SendableSequentialStream, SequentialStreamAdapter,
    SequentialStreamExt as _,
};

pub struct BufferedStrategy {
    child: ArcRef<dyn LayoutStrategy>,
    buffer_size: u64,
}

impl BufferedStrategy {
    pub fn new(child: ArcRef<dyn LayoutStrategy>, buffer_size: u64) -> Self {
        Self { child, buffer_size }
    }
}

impl LayoutStrategy for BufferedStrategy {
    fn write_stream(
        &self,
        ctx: &ArrayContext,
        sequence_writer: SequenceWriter,
        stream: SendableSequentialStream,
    ) -> SendableLayoutWriter {
        let dtype = stream.dtype().clone();
        let buffer_size = self.buffer_size;
        let buffered_stream = try_stream! {
            let stream = stream.peekable();
            pin_mut!(stream);

            let mut nbytes = 0u64;
            let mut chunks = VecDeque::new();

            while let Some(chunk) = stream.as_mut().next().await {
                let (sequence_id, chunk) = chunk?;
                nbytes += chunk.nbytes() as u64;
                chunks.push_back(chunk);

                // if this is the last element, flush everything
                if stream.as_mut().peek().await.is_none() {
                    let mut sequence_pointer = sequence_id.descend();
                    while let Some(chunk) = chunks.pop_front() {
                        yield (sequence_pointer.advance(), chunk)
                    }
                    break;
                }

                if nbytes < 2 * buffer_size {
                    continue;
                };
                // Wait until we're at 2x the buffer size before flushing 1x the buffer size
                // This avoids small tail stragglers being flushed at the end of the file.
                let mut sequence_pointer = sequence_id.descend();
                while nbytes > buffer_size {
                    let Some(chunk) = chunks.pop_front() else {
                        break;
                    };
                    nbytes -= chunk.nbytes() as u64;
                    yield (sequence_pointer.advance(), chunk)
                }
            }
        };
        self.child.write_stream(
            ctx,
            sequence_writer,
            SequentialStreamAdapter::new(dtype, buffered_stream).sendable(),
        )
    }
}
