use std::ops::Deref;

use async_trait::async_trait;
use bytes::Bytes;
use vortex_array::ArrayData;
use vortex_error::VortexResult;
use vortex_ipc::messages::{EncoderMessage, MessageEncoder};

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

#[async_trait]
pub trait AsyncSegmentReader: Send + Sync {
    /// Attempt to get the data associated with a given segment ID.
    ///
    /// If the segment ID is not found, `None` is returned.
    async fn get(&self, id: SegmentId) -> VortexResult<Bytes>;
}

pub trait SegmentWriter {
    /// Write the given data into a segment and return its identifier.
    /// The provided buffers are concatenated together to form the segment.
    fn put(&mut self, data: Vec<Bytes>) -> SegmentId;

    // TODO(ngates): convert this to take an `ArrayParts` so it's obvious to the caller that the
    //  serialized message does not include the array's length or dtype.
    // TODO(ngates): do not use the IPC message encoder since it adds extra unnecessary framing.
    fn put_chunk(&mut self, array: ArrayData) -> SegmentId {
        self.put(MessageEncoder::default().encode(EncoderMessage::Array(&array)))
    }
}

#[cfg(test)]
pub mod test {
    use bytes::{Bytes, BytesMut};
    use vortex_error::{vortex_err, VortexExpect};

    use super::*;

    #[derive(Default)]
    pub struct TestSegments {
        segments: Vec<Bytes>,
    }

    impl SegmentWriter for TestSegments {
        fn put(&mut self, data: Vec<Bytes>) -> SegmentId {
            let id = u32::try_from(self.segments.len())
                .vortex_expect("Cannot store more than u32::MAX segments");
            let mut buffer = BytesMut::with_capacity(data.iter().map(Bytes::len).sum());
            for bytes in data {
                buffer.extend_from_slice(&bytes);
            }
            self.segments.push(buffer.freeze());
            id.into()
        }
    }

    #[async_trait]
    impl AsyncSegmentReader for TestSegments {
        async fn get(&self, id: SegmentId) -> VortexResult<Bytes> {
            self.segments
                .get(*id as usize)
                .cloned()
                .ok_or_else(|| vortex_err!("Segment not found"))
        }
    }
}
