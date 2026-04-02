// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;

use crate::segments::SegmentId;
/// Static future resolving to a segment byte buffer.
pub type SegmentFuture = BoxFuture<'static, VortexResult<BufferHandle>>;

/// Apply a set of segment-relative byte ranges to an already-resolved segment buffer.
pub(crate) fn apply_ranges(
    buffer: BufferHandle,
    ranges: &[Range<usize>],
) -> VortexResult<BufferHandle> {
    match ranges {
        [] => buffer.filter(&[]),
        [range] => Ok(buffer.slice(range.clone())),
        _ => buffer.filter(ranges),
    }
}

/// A trait for providing segment data to a [`crate::LayoutReader`].
pub trait SegmentSource: 'static + Send + Sync {
    /// Return the full length of a segment in bytes if known without issuing I/O.
    fn segment_len(&self, _id: SegmentId) -> Option<usize> {
        None
    }

    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> SegmentFuture;

    /// Request a set of segment-relative byte ranges and return them packed into one contiguous buffer.
    ///
    /// Implementations may satisfy this by issuing multiple underlying reads, but the returned
    /// [`BufferHandle`] must contain the concatenated bytes in the same order as `ranges`.
    ///
    /// Follow-up: push scatter/gather support into the lower I/O API so sources can fill a
    /// caller-owned destination buffer directly and avoid the gather copy that this interface may
    /// currently require.
    fn request_ranges(&self, id: SegmentId, ranges: Vec<Range<usize>>) -> SegmentFuture {
        let future = self.request(id);
        async move {
            let buffer = future.await?;
            apply_ranges(buffer, &ranges)
        }
        .boxed()
    }
}
