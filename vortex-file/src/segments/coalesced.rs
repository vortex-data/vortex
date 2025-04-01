use std::cmp::Ordering;
use std::ops::Range;
use std::sync::Arc;

use vortex_buffer::{Alignment, ByteBuffer};
use vortex_error::{VortexExpect, VortexResult, vortex_panic};
use vortex_io::VortexReadAt;
use vortex_layout::segments::SegmentId;

use crate::SegmentSpec;
use crate::scan::segments::{SegmentEvent, SegmentQueueInner};

pub struct CoalescedSegmentRequest {
    pub(crate) queue: Arc<SegmentQueueInner>,
    /// The alignment of the first segment.
    // TODO(ngates): is this the best alignment to use?
    pub(crate) alignment: Alignment,
    /// The range of the file to read.
    pub(crate) byte_range: Range<u64>,
    /// The original segment requests, ordered by segment ID.
    pub(crate) requests: Vec<SegmentId>,
}

impl CoalescedSegmentRequest {
    fn size_bytes(&self) -> u64 {
        self.byte_range.end - self.byte_range.start
    }
}

pub(crate) async fn evaluate<R: VortexReadAt + Send>(
    read: R,
    request: CoalescedSegmentRequest,
    segment_map: Arc<[SegmentSpec]>,
) -> VortexResult<()> {
    log::debug!(
        "Reading byte range for [{}] requests {:?} size={}",
        request.requests,
        request.byte_range,
        request.byte_range.end - request.byte_range.start,
    );
    let buffer: ByteBuffer = read
        .read_byte_range(request.byte_range.clone(), request.alignment)
        .await?
        .aligned(Alignment::none());

    // Figure out the segments covered by the read.
    let start = segment_map.partition_point(|s| s.offset < request.byte_range.start);
    let end = segment_map.partition_point(|s| s.offset < request.byte_range.end);

    // Note that we may have multiple requests for the same segment.
    let mut requests_iter = request.requests.into_iter().peekable();

    for (i, segment) in segment_map[start..end].iter().enumerate() {
        let segment_id = SegmentId::from(u32::try_from(i + start).vortex_expect("segment id"));
        let offset = usize::try_from(segment.offset - request.byte_range.start)?;
        let buf = buffer
            .slice(offset..offset + segment.length as usize)
            .aligned(segment.alignment);

        // Find any request callbacks and send the buffer
        while let Some(req) = requests_iter.peek() {
            // If the request is before the current segment, we should have already resolved it.
            match req.cmp(&segment_id) {
                Ordering::Less => {
                    // This should never happen, it means we missed a segment request.
                    vortex_panic!("Skipped segment request");
                }
                Ordering::Equal => {
                    // Resolve the request
                    request
                        .queue
                        .submit_event(SegmentEvent::Resolve(segment_id, Ok(buf.clone())));
                }
                Ordering::Greater => {
                    // No request for this segment, so we continue
                    break;
                }
            }
        }
    }

    Ok(())
}
