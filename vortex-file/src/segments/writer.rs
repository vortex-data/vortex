// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::footer::SegmentSpec;
use async_trait::async_trait;
use parking_lot::Mutex;
use std::collections::VecDeque;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexResult, vortex_err};
use vortex_layout::segments::{SegmentId, SegmentSink};
use vortex_layout::sequence::SequenceId;

/// A segment writer that enforces segment id's it receives are monotonically increasing.
/// It does buffer segments in a flush channel.
pub struct FileSegmentWriter {
    state: Mutex<State>,
}

impl FileSegmentWriter {
    pub fn new(byte_offset: u64) -> Self {
        Self {
            state: Mutex::new(State {
                byte_offset,
                buffer_sink: Default::default(),
                segment_specs: Default::default(),
            }),
        }
    }

    /// Drain any buffered segments into the provided sink.
    pub fn drain_to_vec(&self) -> Vec<ByteBuffer> {
        let mut state = self.state.lock();
        state.buffer_sink.drain(..).collect()
    }

    pub fn segment_specs(&self) -> Vec<SegmentSpec> {
        self.state.lock().segment_specs.clone()
    }

    pub fn byte_offset(&self) -> u64 {
        self.state.lock().byte_offset
    }
}

struct State {
    byte_offset: u64,
    buffer_sink: VecDeque<ByteBuffer>,
    segment_specs: Vec<SegmentSpec>,
}

#[async_trait]
impl SegmentSink for FileSegmentWriter {
    async fn write(
        &self,
        sequence_id: SequenceId,
        buffers: Vec<ByteBuffer>,
    ) -> VortexResult<SegmentId> {
        // We wait for all segment IDs before this one to be dropped. Then while we hold this
        // one, we essentially have an exclusive lock on the segment writer. This ensures we
        // don't deadlock while locking the state mutex. We can use a non-async mutex here
        // since we are not performing any async operations while holding the lock.
        let segment_id = sequence_id.collapse().await;
        let mut state = self.state.lock();

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
                .buffer_sink
                .push_back(ByteBuffer::zeroed(padding as usize));
        }
        state.buffer_sink.extend(buffers);

        Ok(segment_id)
    }
}
