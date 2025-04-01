use std::cmp::Ordering;
use std::ops::Range;
use std::sync::Arc;

use futures::Stream;
use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexExpect, VortexResult, vortex_panic};
use vortex_io::VortexReadAt;
use vortex_layout::segments::{SegmentEvents, SegmentId, SegmentRequest};

use crate::SegmentSpec;

pub struct CoalescedSegmentRequest {
    /// The range of the file to read.
    pub(crate) byte_range: Range<u64>,
    /// The original segment requests, ordered by segment ID.
    pub(crate) requests: Vec<SegmentRequest>,
    /// A copy of the segment map so we can resolve the requests.
    pub(crate) segment_map: Arc<[SegmentSpec]>,
}

impl CoalescedSegmentRequest {
    fn size_bytes(&self) -> u64 {
        self.byte_range.end - self.byte_range.start
    }

    /// Resolve the requests with the provided buffer.
    pub fn resolve(self, buffer: VortexResult<ByteBuffer>) {
        let buffer = match buffer {
            Ok(buffer) => buffer,
            Err(e) => {
                // If we fail to read the buffer, we need to resolve all the requests with the error.
                let err = Arc::new(e);
                for request in self.requests {
                    request.resolve(Err(err.clone().into()));
                }
                return;
            }
        };

        if buffer.len() != self.size_bytes() as usize {
            vortex_panic!(
                "Buffer size mismatch: expected {} bytes, got {}",
                self.size_bytes(),
                buffer.len()
            );
        }

        // Split the buffer into segments and resolve the requests.
        for request in self.requests {
            let spec = &self.segment_map[*request.id() as usize];
            let start = usize::try_from(spec.offset - self.byte_range.start)
                .vortex_expect("start too large");
            let stop =
                usize::try_from(start + spec.length as usize).vortex_expect("length too large");
            request.resolve(Ok(buffer.slice(start..stop).aligned(spec.alignment)))
        }
    }
}
