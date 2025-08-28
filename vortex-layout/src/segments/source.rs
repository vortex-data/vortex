// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::segments::SegmentId;
use futures::future::BoxFuture;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_io::runtime::Handle;

/// Future resolving to a segment byte buffer that depends only on the runtime.
pub type SegmentFuture<'handle> = BoxFuture<'handle, VortexResult<ByteBuffer>>;

/// A trait for providing segment data to a [`crate::LayoutReader`].
pub trait SegmentSource: 'static + Send + Sync {
    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request<'handle>(&self, id: SegmentId, handle: &Handle<'handle>) -> SegmentFuture<'handle>;
}
