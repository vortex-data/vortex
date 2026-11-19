// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use futures::FutureExt;
use parking_lot::Mutex;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::ByteBuffer;
use vortex_buffer::ByteBufferMut;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::segments::SegmentFuture;
use crate::segments::SegmentId;
use crate::segments::SegmentSink;
use crate::segments::SegmentSource;
use crate::sequence::SequenceId;

/// A dummy in-memory implementation of a segment reader and writer.
#[derive(Default, Clone)]
pub struct TestSegments {
    segments: Arc<Mutex<Vec<ByteBuffer>>>,
}

impl SegmentSource for TestSegments {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        let buffer = self.segments.lock().get(*id as usize).cloned();
        async move {
            buffer
                .map(BufferHandle::new_host)
                .ok_or_else(|| vortex_err!("Segment not found"))
        }
        .boxed()
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

/// A [`SegmentSource`] that records the segments requested from an inner source.
///
/// Clones share the record.
#[derive(Clone)]
pub struct CountingSegmentSource {
    inner: Arc<dyn SegmentSource>,
    requested: Arc<Mutex<Vec<SegmentId>>>,
}

impl CountingSegmentSource {
    /// Wraps `inner`, with nothing requested yet.
    pub fn new(inner: Arc<dyn SegmentSource>) -> Self {
        Self {
            inner,
            requested: Arc::default(),
        }
    }

    /// Returns how many segments have been requested so far.
    pub fn requests(&self) -> usize {
        self.requested.lock().len()
    }

    /// Returns the segments requested so far, in request order.
    pub fn requested(&self) -> Vec<SegmentId> {
        self.requested.lock().clone()
    }
}

impl SegmentSource for CountingSegmentSource {
    fn request(&self, id: SegmentId) -> SegmentFuture {
        self.requested.lock().push(id);
        self.inner.request(id)
    }
}
