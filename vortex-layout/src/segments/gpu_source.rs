// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use futures::future::BoxFuture;
use vortex_error::VortexResult;
use vortex_gpu::CudaByteBuffer;

use crate::segments::SegmentId;

pub type GpuSegmentFuture = BoxFuture<'static, VortexResult<CudaByteBuffer>>;

/// A trait for providing segment data to a [`crate::LayoutReader`].
pub trait GpuSegmentSource: 'static + Send + Sync {
    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> GpuSegmentFuture;
}
