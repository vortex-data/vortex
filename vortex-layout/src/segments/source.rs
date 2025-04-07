use std::sync::Arc;

use futures::future::BoxFuture;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::segments::SegmentId;

/// Static future resolving to a segment byte buffer.
pub type SegmentFuture = BoxFuture<'static, VortexResult<ByteBuffer>>;

/// A trait for providing segment data to a [`crate::LayoutReader`].
pub trait SegmentSource: 'static + Send + Sync {
    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId, for_whom: &Arc<str>) -> SegmentFuture;
}
