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

/// A trait for providing segment data to a [`crate::LayoutReader`].
pub trait SegmentSource: 'static + Send + Sync {
    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> SegmentFuture;

    /// Request a byte range within a segment.
    ///
    /// The default implementation reads the full segment and then slices.
    /// Implementations backed by random-access storage (files, object stores) should override
    /// this to issue a targeted read.
    fn request_range(&self, id: SegmentId, range: Range<usize>) -> SegmentFuture {
        let fut = self.request(id);
        async move {
            let buffer = fut.await?;
            let host = buffer.try_to_host_sync()?;
            Ok(BufferHandle::new_host(host.slice_unaligned(range)))
        }
        .boxed()
    }
}
