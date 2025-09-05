// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt;
use parking_lot::Mutex;
use vortex_buffer::{ByteBuffer, ByteBufferMut};
use vortex_error::{vortex_err, VortexExpect, VortexResult};

use crate::segments::{SegmentFuture, SegmentId, SegmentSink, SegmentSource};
use crate::sequence::SequenceId;

/// A dummy in-memory implementation of a segment reader and writer.
#[derive(Default, Clone)]
pub struct TestSegments {
    segments: Arc<Mutex<Vec<ByteBuffer>>>,
}

impl SegmentSource for TestSegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let buffer = self.segments.lock().get(*id as usize).cloned();
        async move { buffer.ok_or_else(|| vortex_err!("Segment not found")) }.boxed()
    }
}

#[async_trait]
impl SegmentSink for TestSegments {
    async fn write(
        &self,
        _sequence_id: SequenceId,
        buffers: Vec<ByteBuffer>,
    ) -> VortexResult<SegmentId> {
        // Combine all the buffers since we're only a test implementation
        let mut buffer = ByteBufferMut::empty();
        for segment in buffers {
            buffer.extend_from_slice(segment.as_ref());
        }

        let mut segments = self.segments.lock();
        let segment_id =
            SegmentId::from(u32::try_from(segments.len()).vortex_expect("Too many segments"));
        segments.push(buffer.freeze());

        Ok(segment_id)
    }
}
