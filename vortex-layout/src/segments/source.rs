// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use futures::future::BoxFuture;
use vortex_array::{ArrayContext, ArrayRef};
use vortex_buffer::ByteBuffer;
use vortex_dtype::DType;
use vortex_error::VortexResult;

use crate::segments::SegmentId;

/// Static future resolving to a segment byte buffer.
pub type SegmentFuture = BoxFuture<'static, VortexResult<ByteBuffer>>;

/// A trait for providing segment data to a [`crate::LayoutReader`].
pub trait SegmentSource: 'static + Send + Sync {
    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> SegmentFuture;

    fn array_cache(self: Arc<Self>) -> Option<Arc<dyn ArrayCache>> {
        None
    }
}

pub type ArrayFuture<'a> = BoxFuture<'a, VortexResult<ArrayRef>>;

pub trait ArrayCache: Send + Sync {
    fn get<'a>(
        &'a self,
        segment_id: SegmentId,
        ctx: ArrayContext,
        dtype: DType,
        len: usize,
    ) -> ArrayFuture<'a>;
}
