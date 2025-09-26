// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use async_trait::async_trait;
use parking_lot::Mutex;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexResult, vortex_err};
use vortex_layout::segments::{SegmentId, SegmentSink};
use vortex_layout::sequence::SequenceId;

use crate::footer::SegmentSpec;

pub struct BufferedSegmentSink {
    buffers: kanal::AsyncSender<ByteBuffer>,
    byte_offset: AtomicU64,
    segment_specs: Mutex<Vec<SegmentSpec>>,
}

impl BufferedSegmentSink {
    pub fn new(send: kanal::AsyncSender<ByteBuffer>, byte_offset: u64) -> Self {
        Self {
            buffers: send,
            byte_offset: AtomicU64::new(byte_offset),
            segment_specs: Default::default(),
        }
    }

    /// Close the sink, returning the segment specs and the final byte offset.
    pub fn segment_specs(&self) -> Arc<[SegmentSpec]> {
        let specs = self.segment_specs.lock();
        specs.clone().into()
    }
}

#[async_trait]
impl SegmentSink for BufferedSegmentSink {
    async fn write(
        &self,
        mut sequence_id: SequenceId,
        buffers: Vec<ByteBuffer>,
    ) -> VortexResult<SegmentId> {
        // We wait for all segment IDs before this one to be dropped. Then while we hold a strong
        // reference to this one, we essentially have an exclusive lock on the segment writer.
        sequence_id.collapse().await;

        let (segment_id, padding_buffer) = {
            let mut specs = self.segment_specs.lock();
            let segment_id = SegmentId::from(
                u32::try_from(specs.len())
                    .map_err(|_| vortex_err!("Too mant segments, u32 overflow"))?,
            );

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
            let byte_offset = self.byte_offset.load(Ordering::Relaxed);
            let padding = byte_offset.next_multiple_of(*alignment as u64) - byte_offset;
            let offset = byte_offset + padding;
            specs.push(SegmentSpec {
                offset,
                length,
                alignment,
            });

            self.byte_offset
                .store(byte_offset + padding + u64::from(length), Ordering::Relaxed);

            // Send the buffers to the stream.
            if padding > 0 {
                (segment_id, Some(ByteBuffer::zeroed(padding as usize)))
            } else {
                (segment_id, None)
            }
        };

        if let Some(padding) = padding_buffer {
            let _ = self.buffers.send(padding).await;
        }
        for buffer in buffers {
            let _ = self.buffers.send(buffer).await;
        }

        Ok(segment_id)
    }
}
