// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::segments::SegmentId;
/// Static future resolving to a segment byte buffer.
pub type SegmentFuture = BoxFuture<'static, VortexResult<BufferHandle>>;

/// Provides segment data to a [`crate::LayoutReader`].
///
/// Implementations may issue asynchronous file reads, object-store requests, cache lookups, or
/// in-memory buffer slices. Returned futures must be independent and safe to poll concurrently.
pub trait SegmentSource: 'static + Send + Sync {
    /// Preferred size of independently requested byte ranges for this source.
    ///
    /// Layout readers can use this hint to divide a logical segment into canonical read ranges.
    /// Returning `None` asks readers to preserve whole-segment reads.
    fn preferred_read_size(&self) -> Option<u64> {
        None
    }

    /// Return the serialized length of `id`, when it is known without issuing I/O.
    fn segment_len(&self, _id: SegmentId) -> Option<u64> {
        None
    }

    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> SegmentFuture;

    /// Request a byte range relative to the start of a segment.
    ///
    /// Sources backed by random-access storage should override this method. The default keeps
    /// custom sources compatible by reading the segment and slicing it after bounds checking.
    fn request_range(&self, id: SegmentId, range: Range<u64>) -> SegmentFuture {
        let segment = self.request(id);
        async move {
            let segment = segment.await?;
            let start = usize::try_from(range.start)?;
            let end = usize::try_from(range.end)?;
            if start > end || end > segment.len() {
                vortex_bail!(
                    "Segment {} range {}..{} is out of bounds for a {}-byte segment",
                    id,
                    range.start,
                    range.end,
                    segment.len()
                );
            }
            Ok(segment.slice(start..end))
        }
        .boxed()
    }

    /// Register multiple ranges from one segment together.
    ///
    /// The returned futures correspond positionally to `ranges`. Sources can override this to
    /// amortize registration while retaining independent canonical range futures for sharing and
    /// coalescing.
    fn request_ranges(&self, id: SegmentId, ranges: Vec<Range<u64>>) -> Vec<SegmentFuture> {
        ranges
            .into_iter()
            .map(|range| self.request_range(id, range))
            .collect()
    }
}
