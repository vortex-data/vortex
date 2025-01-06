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
pub mod test {
    use std::sync::Arc;

    use bytes::{Bytes, BytesMut};
    use vortex_error::{vortex_panic, VortexExpect};

    use super::*;
    use crate::scanner::{LayoutScan, Poll};
    use crate::segments::SegmentReader;
    use crate::RowMask;

    #[derive(Default)]
    pub struct TestSegments {
        segments: Vec<Bytes>,
    }

    impl TestSegments {
        pub fn do_scan(&self, scan: Arc<dyn LayoutScan>) -> ArrayData {
            let row_count = scan.layout().row_count();
            let mut scanner = scan
                .create_scanner(RowMask::new_valid_between(0, row_count))
                .vortex_expect("Failed to create scanner");
            match scanner.poll(self).vortex_expect("Failed to poll scanner") {
                Poll::Some(array) => array,
                Poll::NeedMore(_segments) => {
                    vortex_panic!("Layout requested more segments from TestSegments.")
                }
            }
        }
    }

    impl SegmentReader for TestSegments {
        fn get(&self, id: SegmentId) -> Option<Bytes> {
            self.segments.get(*id as usize).cloned()
        }
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
}
