// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::collections::VecDeque;

use async_trait::async_trait;
use parking_lot::Mutex;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexResult, vortex_err};
use vortex_layout::segments::{SegmentId, SegmentSink};
use vortex_layout::sequence::SequenceId;

use crate::footer::SegmentSpec;

pub struct BufferedSegmentSink {
    state: Mutex<State>,
}

struct State {
    byte_offset: u64,
    buffers: VecDeque<ByteBuffer>,
    segment_specs: Vec<SegmentSpec>,
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

        // The sequence ID essentially gives us a lock, so the mutex here is never contended.
        let mut state = self.state.lock();

        let segment_id = SegmentId::from(
            u32::try_from(state.segment_specs.len())
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
        let padding = state.byte_offset.next_multiple_of(*alignment as u64) - state.byte_offset;
        let offset = state.byte_offset + padding;
        state.segment_specs.push(SegmentSpec {
            offset,
            length,
            alignment,
        });
        state.byte_offset += padding + u64::from(length);

        // Send the buffers to the stream.
        if padding > 0 {
            state
                .buffers
                .push_back(ByteBuffer::zeroed(padding as usize));
        }
        state.buffers.extend(buffers);

        Ok(segment_id)
    }
}

impl BufferedSegmentSink {
    pub fn new(initial_buffers: impl IntoIterator<Item = ByteBuffer>) -> Self {
        let mut buffers = VecDeque::new();
        let mut byte_offset = 0;
        for buffer in initial_buffers {
            byte_offset += buffer.len() as u64;
            buffers.push_back(buffer);
        }
        Self {
            state: Mutex::new(State {
                byte_offset,
                buffers,
                segment_specs: Vec::new(),
            }),
        }
    }

    /// Drain the internal buffers to a vector.
    pub fn drain_to_vec(&self) -> Vec<ByteBuffer> {
        let mut state = self.state.lock();
        state.buffers.drain(..).collect()
    }

    pub fn byte_offset(&self) -> u64 {
        let state = self.state.lock();
        state.byte_offset
    }

    pub fn into_specs(self) -> Vec<SegmentSpec> {
        let state = self.state.lock();
        state.segment_specs.clone()
    }
}
