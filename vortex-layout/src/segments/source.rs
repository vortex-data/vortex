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
        [] => buffer.copy_ranges(&[]),
        [range] if range.start.is_multiple_of(*buffer.alignment()) => {
            Ok(buffer.slice(range.clone()))
        }
        [range] => buffer.copy_ranges(std::slice::from_ref(range)),
        _ => buffer.copy_ranges(ranges),
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
    ///
    /// The intended split is:
    /// - this API continues to describe the logical bytes to materialize;
    /// - sources may normalize/merge ranges before issuing physical reads;
    /// - lower I/O backends may optionally expose vectored reads into output slices.
    ///
    /// That lets local files eventually use `preadv`/`preadv2`-style reads for sparse ranges,
    /// while remote/object-store backends can still choose a smaller number of coalesced
    /// contiguous reads when that is cheaper. A later follow-up should also thread through
    /// alignment requirements for `DIRECT_IO`, since that may force padded physical reads even
    /// when the logical request stays sparse.
    fn request_ranges(&self, id: SegmentId, ranges: Vec<Range<usize>>) -> SegmentFuture {
        let future = self.request(id);
        async move {
            let buffer = future.await?;
            apply_ranges(buffer, &ranges)
        }
        .boxed()
    }
}
