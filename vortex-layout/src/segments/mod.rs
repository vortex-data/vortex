use std::ops::Deref;

use bytes::Bytes;
use vortex_array::ArrayData;
use vortex_ipc::messages::{EncoderMessage, MessageEncoder};

/// The identifier for a single segment.
// TODO(ngates): should this be a `[u8]` instead? Allowing for arbitrary segment identifiers?
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
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

pub trait SegmentReader {
    /// Attempt to get the data associated with a given segment ID.
    ///
    /// If the segment ID is not found, `None` is returned.
    fn get(&self, id: SegmentId) -> Option<Bytes>;
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
mod test {
    use bytes::{Bytes, BytesMut};

    use super::*;
    use crate::segments::SegmentReader;

    #[derive(Default)]
    pub struct TestSegments {
        segments: Vec<Bytes>,
    }

    impl SegmentReader for TestSegments {
        fn get(&self, id: SegmentId) -> Option<Bytes> {
            self.segments.get(*id as usize).cloned()
        }
    }

    impl SegmentWriter for TestSegments {
        fn put(&mut self, data: Vec<Bytes>) -> SegmentId {
            let id = self.segments.len() as u32;
            let mut buffer = BytesMut::with_capacity(data.iter().map(Bytes::len).sum());
            for bytes in data {
                buffer.extend_from_slice(&bytes);
            }
            self.segments.extend(buffer.freeze());
            id.into()
        }
    }
}
