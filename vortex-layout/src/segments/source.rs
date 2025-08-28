// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use crate::segments::SegmentId;
use futures::future::BoxFuture;
use std::sync::Arc;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

/// Future resolving to a segment byte buffer that depends only on the runtime.
pub type SegmentFuture<'rt> = BoxFuture<'rt, VortexResult<ByteBuffer>>;

pub type SegmentSourceRef<'rt> = Arc<dyn SegmentSource<'rt> + 'rt>;

/// A trait for providing segment data to a [`crate::LayoutReader`].
pub trait SegmentSource<'rt>: 'rt + Send + Sync {
    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> SegmentFuture<'rt>;
}
