use std::sync::Arc;

use futures::FutureExt;
use vortex_buffer::{ByteBuffer, ByteBufferMut};
use vortex_error::{VortexExpect, vortex_err};

use crate::segments::sink::SegmentWriter;
use crate::segments::{SegmentFuture, SegmentId, SegmentSource};

/// A dummy in-memory implementation of a segment reader and writer.
#[derive(Default)]
pub struct TestSegments {
    segments: Vec<ByteBuffer>,
}

impl SegmentSource for TestSegments {
    fn request(&self, id: SegmentId, _for_whom: &Arc<str>) -> SegmentFuture {
        let buffer = self.segments.get(*id as usize).cloned();
        async move { buffer.ok_or_else(|| vortex_err!("Segment not found")) }.boxed()
    }
}

impl SegmentWriter for TestSegments {
    fn put(&mut self, data: &[ByteBuffer]) -> SegmentId {
        let id = u32::try_from(self.segments.len())
            .vortex_expect("Cannot store more than u32::MAX segments");

        // Combine all the buffers since we're only a test implementation
        let mut buffer = ByteBufferMut::empty();
        for segment in data {
            buffer.extend_from_slice(segment.as_ref());
        }
        self.segments.push(buffer.freeze());

        id.into()
    }
}
