// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use futures::channel::mpsc;
use futures::{SinkExt, stream};
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexResult, vortex_bail, vortex_err};
use vortex_layout::segments::{SegmentId, SegmentWriter};

use crate::footer::SegmentSpec;

/// A segment writer that enforces segment id's it receives are monotonically increasing.
/// It does buffer segments in a flush channel.
pub struct SerialSegmentWriter {
    buffer_sink: mpsc::Sender<ByteBuffer>,
    next_expected: SegmentId,
    byte_offset: u64,
    segment_specs: Vec<SegmentSpec>,
}

impl SerialSegmentWriter {
    pub fn into_parts(self) -> (u64, Vec<SegmentSpec>) {
        (self.byte_offset, self.segment_specs)
    }
}

#[async_trait]
impl SegmentWriter for SerialSegmentWriter {
    async fn put(&mut self, segment_id: SegmentId, buffers: Vec<ByteBuffer>) -> VortexResult<()> {
        if segment_id != self.next_expected {
            vortex_bail!(
                "out of order segment id, expected {:?}, got {:?}",
                self.next_expected,
                segment_id
            );
        }
        self.next_expected = SegmentId::from(*segment_id + 1);

        // The API requires us to write these buffers contiguously. Therefore, we can only
        // respect the alignment of the first one.
        // Don't worry, in most cases the caller knows what they're doing and will align the
        // buffers themselves, inserting padding buffers where necessary.
        let alignment = buffers
            .first()
            .map(|buffer| buffer.alignment())
            .unwrap_or_else(Alignment::none);
        let length = u32::try_from(buffers.iter().map(|buffer| buffer.len()).sum::<usize>())
            .map_err(|_| vortex_err!("segment buffer length exceeds maximum u32"))?;

        // Add any padding required to align the segment.
        let padding = self.byte_offset.next_multiple_of(*alignment as u64) - self.byte_offset;
        self.segment_specs.push(SegmentSpec {
            offset: self.byte_offset + padding,
            length,
            alignment,
        });
        self.byte_offset += padding + u64::from(length);

        // Send the buffers to the stream.
        if padding > 0 {
            self.buffer_sink
                .send(ByteBuffer::zeroed(padding as usize))
                .await
                .map_err(|_| vortex_err!("failed to send padding buffer to flusher"))?;
        }
        self.buffer_sink
            .send_all(&mut stream::iter(buffers))
            .await
            .map_err(|_| vortex_err!("failed to send segment buffers to flusher"))?;

        Ok(())
    }
}

impl SerialSegmentWriter {
    /// Create a [SegmentWriter] and a [SegmentFlusher].
    pub fn create(initial_offset: u64, buffer_sink: mpsc::Sender<ByteBuffer>) -> Self {
        SerialSegmentWriter {
            buffer_sink,
            next_expected: SegmentId::from(0),
            byte_offset: initial_offset,
            segment_specs: Vec::new(),
        }
    }
}
