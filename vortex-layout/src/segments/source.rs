// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::ops::Range;

use futures::FutureExt;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

use crate::segments::SegmentId;
/// Static future resolving to a segment byte buffer.
pub type SegmentFuture = BoxFuture<'static, VortexResult<BufferHandle>>;

/// Provides segment data to a [`crate::LayoutReader`].
///
/// Implementations may issue asynchronous file reads, object-store requests, cache lookups, or
/// in-memory buffer slices. Returned futures must be independent and safe to poll concurrently.
pub trait SegmentSource: 'static + Send + Sync {
    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> SegmentFuture;

    /// Whether range requests can avoid fetching the whole segment.
    fn supports_ranges(&self) -> bool {
        false
    }

    /// Requests a byte range relative to a segment, without admitting a partial segment to caches.
    /// The default implementation reads the segment and slices it.
    fn request_range(&self, id: SegmentId, range: Range<usize>) -> SegmentFuture {
        let read = self.request(id);
        async move {
            let bytes = read.await?;
            vortex_ensure!(
                range.start <= range.end && range.end <= bytes.len(),
                "Segment range out of bounds"
            );
            Ok(bytes.slice(range))
        }
        .boxed()
    }
}
