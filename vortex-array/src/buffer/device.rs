// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;

use futures::future::BoxFuture;
use vortex_buffer::Alignment;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_utils::dyn_traits::DynEq;
use vortex_utils::dyn_traits::DynHash;

/// A buffer that is stored on the GPU.
pub trait DeviceBuffer: 'static + Send + Sync + Debug + DynEq + DynHash {
    /// Returns a reference as `Any` to enable downcasting.
    fn as_any(&self) -> &dyn Any;

    /// Returns the length of the buffer in bytes.
    fn len(&self) -> usize;

    /// Returns the alignment of the buffer.
    fn alignment(&self) -> Alignment;

    /// Returns true if the buffer is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Attempts to copy the device buffer to a host ByteBuffer.
    ///
    /// # Errors
    ///
    /// This operation may fail, depending on the device implementation and the underlying hardware.
    fn copy_to_host_sync(&self, alignment: Alignment) -> VortexResult<ByteBuffer>;

    /// Copies the device buffer to a host buffer asynchronously.
    ///
    /// Schedules an async copy and returns a future that completes when the copy is finished.
    ///
    /// # Arguments
    ///
    /// * `alignment` - The memory alignment to use for the host buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the async copy operation fails.
    fn copy_to_host(
        &self,
        alignment: Alignment,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ByteBuffer>>>;

    /// Create a new buffer that references a subrange of this buffer at the given
    /// slice indices.
    ///
    /// Note that slice indices are in byte units.
    fn slice(&self, range: Range<usize>) -> Arc<dyn DeviceBuffer>;

    /// Return a buffer with the given alignment. Where possible, this will be zero-copy.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer cannot be aligned (e.g., allocation or copy failure).
    fn aligned(self: Arc<Self>, alignment: Alignment) -> VortexResult<Arc<dyn DeviceBuffer>>;
}

pub trait DeviceBufferExt: DeviceBuffer {
    /// Slice a range of elements `T` out of the device buffer.
    fn slice_typed<T: Sized>(&self, range: Range<usize>) -> Arc<dyn DeviceBuffer>;
}

impl<B: DeviceBuffer> DeviceBufferExt for B {
    fn slice_typed<T: Sized>(&self, range: Range<usize>) -> Arc<dyn DeviceBuffer> {
        let start_bytes = range.start * size_of::<T>();
        let end_bytes = range.end * size_of::<T>();
        self.slice(start_bytes..end_bytes)
    }
}

impl Hash for dyn DeviceBuffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.dyn_hash(state);
    }
}

impl PartialEq for dyn DeviceBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.dyn_eq(other)
    }
}
impl Eq for dyn DeviceBuffer {}
