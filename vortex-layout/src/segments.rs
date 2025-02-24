use std::fmt::Display;
use std::ops::Deref;

use async_trait::async_trait;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

/// The identifier for a single segment.
// TODO(ngates): should this be a `[u8]` instead? Allowing for arbitrary segment identifiers?
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SegmentId(u32);

impl From<u32> for SegmentId {
    fn from(value: u32) -> Self {
        Self(value)
    }
}

impl Deref for SegmentId {
    type Target = u32;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl Display for SegmentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "SegmentId({})", self.0)
    }
}

#[async_trait]
pub trait AsyncSegmentReader: 'static + Send + Sync {
    /// Attempt to get the data associated with a given segment ID.
    async fn get(&self, id: SegmentId) -> VortexResult<ByteBuffer>;
}

pub trait SegmentWriter {
    /// Write the given data into a segment and return its identifier.
    /// The provided buffers are concatenated together to form the segment.
    ///
    // TODO(ngates): in order to support aligned Direct I/O, it is preferable for all segments to
    //  be aligned to the logical block size (typically 512, but could be 4096). For this reason,
    //  if we know we're going to read an entire FlatLayout together, then we should probably
    //  serialize it into a single segment that is 512 byte aligned? Or else, we should guarantee
    //  to align the the first segment to 512, and then assume that coalescing captures the rest.
    fn put(&mut self, buffer: &[ByteBuffer]) -> SegmentId;
}

#[cfg(test)]
pub mod test {
    use vortex_buffer::ByteBufferMut;
    use vortex_error::{VortexExpect, vortex_err};

    use super::*;

    #[derive(Default)]
    pub struct TestSegments {
        segments: Vec<ByteBuffer>,
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

    #[async_trait]
    impl AsyncSegmentReader for TestSegments {
        async fn get(&self, id: SegmentId) -> VortexResult<ByteBuffer> {
            self.segments
                .get(*id as usize)
                .cloned()
                .ok_or_else(|| vortex_err!("Segment not found"))
        }
    }
}
