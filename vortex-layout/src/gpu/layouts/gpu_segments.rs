// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use cudarc::driver::CudaSlice;
use futures::future::BoxFuture;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;

use crate::segments::SegmentId;

/// Static future resolving to a segment byte buffer.
pub type GpuSegmentFuture = BoxFuture<'static, VortexResult<CudaSlice>>;

/// A trait for providing segment data to a [`crate::LayoutReader`].
pub trait GpuSegmentSource: 'static + Send + Sync {
    /// Request a segment, returning a future that will eventually resolve to the segment data.
    fn request(&self, id: SegmentId) -> GpuSegmentFuture;
}
