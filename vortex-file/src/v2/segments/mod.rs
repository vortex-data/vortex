mod cache;
pub(crate) mod channel;
pub(crate) mod writer;

pub use cache::*;
use futures::channel::oneshot;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_layout::segments::SegmentId;

pub struct SegmentRequest {
    // The ID of the requested segment
    pub id: SegmentId,
    // The one-shot channel to send the segment back to the caller
    pub callback: oneshot::Sender<VortexResult<ByteBuffer>>,
}
