mod cache;
pub(crate) mod channel;
pub(crate) mod writer;

pub use cache::*;
use oneshot;
use vortex_buffer::ByteBuffer;
use vortex_error::{VortexExpect, VortexResult, vortex_err};
use vortex_layout::segments::SegmentId;

#[derive(Debug)]
pub struct SegmentRequest {
    // The ID of the requested segment
    pub id: SegmentId,
    // The one-shot channel to send the segment back to the caller
    pub callback: oneshot::Sender<VortexResult<ByteBuffer>>,
}

impl SegmentRequest {
    pub fn resolve(self, buffer: VortexResult<ByteBuffer>) {
        self.callback
            .send(buffer)
            .map_err(|_| vortex_err!("send failed"))
            .vortex_expect("send failed");
    }
}
