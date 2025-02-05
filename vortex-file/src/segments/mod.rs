mod cache;
pub(crate) mod channel;
mod file_source;
mod source;
pub(crate) mod writer;

pub use cache::*;
use futures::channel::oneshot;
pub use source::*;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_layout::segments::SegmentId;

#[derive(Debug)]
pub struct SegmentRequest {
    // The ID of the requested segment
    pub id: SegmentId,
    // The one-shot channel to send the segment back to the caller
    pub callback: oneshot::Sender<VortexResult<ByteBuffer>>,
}
