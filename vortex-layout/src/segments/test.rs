// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use crate::segments::sink::SegmentWriter;
use crate::segments::{SegmentFuture, SegmentId, SegmentSource};
use parking_lot::Mutex;
use vortex_buffer::{ByteBuffer, ByteBufferMut};
use vortex_error::{vortex_err, VortexResult};

/// A dummy in-memory implementation of a segment reader and writer.
#[derive(Default, Clone)]
pub struct TestSegments {
    segments: Arc<Mutex<Vec<ByteBuffer>>>,
}

impl SegmentSource<'_> for TestSegments {
    fn request(&self, id: SegmentId) -> VortexResult<SegmentFuture<'static>> {
        let buffer = self
            .segments
            .lock()
            .get(*id as usize)
            .cloned()
            .ok_or_else(|| vortex_err!("Segment {} not found", id))?;
        Ok(SegmentFuture::new(buffer.len(), async move { Ok(buffer) }))
    }
}

impl SegmentWriter for TestSegments {
    fn put(&mut self, _segment_id: SegmentId, data: Vec<ByteBuffer>) -> VortexResult<()> {
        // Combine all the buffers since we're only a test implementation
        let mut buffer = ByteBufferMut::empty();
        for segment in data {
            buffer.extend_from_slice(segment.as_ref());
        }
        self.segments.lock().push(buffer.freeze());
        Ok(())
    }
}
