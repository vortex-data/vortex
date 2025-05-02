use async_trait::async_trait;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::segments::SegmentId;

#[async_trait]
pub trait SegmentWriter: Send {
    /// Write the given data into a segment and return its identifier.
    /// The provided buffers are concatenated together to form the segment.
    ///
    // TODO(ngates): in order to support aligned Direct I/O, it is preferable for all segments to
    //  be aligned to the logical block size (typically 512, but could be 4096). For this reason,
    //  if we know we're going to read an entire FlatLayout together, then we should probably
    //  serialize it into a single segment that is 512 byte aligned? Or else, we should guarantee
    //  to align the the first segment to 512, and then assume that coalescing captures the rest.
    async fn put(&mut self, buffer: Vec<ByteBuffer>) -> VortexResult<SegmentId>;
}

pub trait ConcurrentSegmentWriter: SegmentWriter {
    /// Splits this writer into multiple writers that maintain a sequential ordering guarantee.
    ///
    /// Creates `splits` additional writers, returning them in a vector. The original writer
    /// is modified to become the last writer in the sequence. This guarantees that segments
    /// written to writers with lower indices will be processed before segments written to
    /// writers with higher indices, with the original writer processing its segments last.
    fn split_off(&mut self, splits: usize) -> VortexResult<Vec<Box<dyn ConcurrentSegmentWriter>>>;
}
