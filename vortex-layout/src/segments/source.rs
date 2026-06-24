// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::segments::SegmentId;
use crate::segments::SegmentInfo;

/// Static future resolving to a segment byte buffer.
pub type SegmentFuture = BoxFuture<'static, VortexResult<BufferHandle>>;

/// Provides segment data to a [`crate::LayoutReader`].
///
/// Implementations may issue asynchronous file reads, object-store requests, cache lookups, or
/// in-memory buffer slices. Returned futures must be independent and safe to poll concurrently.
pub trait SegmentSource: 'static + Send + Sync {
    /// Return scheduler-visible metadata for a segment.
    fn segment_info(&self, id: SegmentId) -> VortexResult<SegmentInfo>;

    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> SegmentFuture;

    /// Return a segment that has already been resolved by the scan scheduler.
    fn resolved(&self, id: SegmentId) -> VortexResult<BufferHandle> {
        vortex_bail!("segment {id} has not been resolved by the scan scheduler")
    }
}
