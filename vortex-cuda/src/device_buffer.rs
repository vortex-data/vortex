// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::CudaView;
use cudarc::driver::DevicePtr;
use cudarc::driver::DeviceRepr;
use vortex_array::buffer::BufferHandle;
use vortex_array::buffer::DeviceBuffer;
use vortex_buffer::Alignment;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

/// A CUDA device buffer with offset and length tracking.
pub struct CudaDeviceBuffer<T> {
    inner: Arc<CudaSlice<T>>,
    offset: usize,
    len: usize,
    device_ptr: u64,
}

impl<T: DeviceRepr> CudaDeviceBuffer<T> {
    /// Creates a new CUDA device buffer from a [`CudaSlice`].
    pub fn new(cuda_slice: CudaSlice<T>) -> Self {
        let len = cuda_slice.len();
        let device_ptr = cuda_slice.device_ptr(cuda_slice.stream()).0;

        Self {
            inner: Arc::new(cuda_slice),
            offset: 0,
            len,
            device_ptr,
        }
    }

    /// Returns a [`CudaView`] to the CUDA device buffer.
    pub fn as_view(&self) -> CudaView<'_, T> {
        self.inner.slice(self.offset..self.offset + self.len)
    }
}

/// Extension trait for getting a [`CudaView`] from a [`BufferHandle`].
pub trait CudaBufferExt {
    /// Returns a [`CudaView`] for the buffer handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is not on the device.
    fn cuda_view<T: DeviceRepr + Send + Sync + 'static>(&self) -> VortexResult<CudaView<'_, T>>;
}

impl CudaBufferExt for BufferHandle {
    fn cuda_view<T: DeviceRepr + Send + Sync + 'static>(&self) -> VortexResult<CudaView<'_, T>> {
        let device_buffer = self
            .as_device_opt()
            .ok_or_else(|| vortex_err!("Buffer is not on device"))?;

        let cuda_buf = device_buffer
            .as_any()
            .downcast_ref::<CudaDeviceBuffer<T>>()
            .ok_or_else(|| {
                vortex_err!(
                    "Device buffer is not a CUDA device buffer for type {}",
                    std::any::type_name::<T>()
                )
            })?;

        Ok(cuda_buf.as_view())
    }
}

impl<T: DeviceRepr> Debug for CudaDeviceBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDeviceBuffer")
            .field("device_ptr", &self.device_ptr)
            .field("offset", &self.offset)
            .field("len", &self.len)
            .finish()
    }
}

impl<T: DeviceRepr> std::hash::Hash for CudaDeviceBuffer<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.device_ptr.hash(state);
        self.len.hash(state);
        self.offset.hash(state);
    }
}

impl<T: DeviceRepr> PartialEq for CudaDeviceBuffer<T> {
    fn eq(&self, other: &Self) -> bool {
        self.device_ptr == other.device_ptr && self.len == other.len && self.offset == other.offset
    }
}

impl<T: DeviceRepr + Send + Sync + 'static> DeviceBuffer for CudaDeviceBuffer<T> {
    /// Returns the number of elements in the buffer of type T.
    fn len(&self) -> usize {
        self.len
    }

    /// Synchronous copy of CUDA device to host memory.
    ///
    /// # Arguments
    ///
    /// * `alignment` - The memory alignment to use for the host buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA memory copy operation fails.
    fn copy_to_host(&self, alignment: Alignment) -> VortexResult<ByteBuffer> {
        let mut host_buffer = BufferMut::<T>::with_capacity_aligned(self.len, alignment);

        let view = self.as_view();
        self.inner
            .stream()
            // TODO(0ax1): make the copy async
            .memcpy_dtoh(&view, unsafe {
                host_buffer.set_len(self.len);
                host_buffer.as_mut_slice()
            })
            .map_err(|e| vortex_err!("Failed to copy from device to host: {}", e))?;

        Ok(host_buffer.freeze().into_byte_buffer())
    }

    /// Slices the CUDA device buffer to a subrange.
    fn slice(&self, range: Range<usize>) -> Arc<dyn DeviceBuffer> {
        let new_offset = self.offset + range.start;
        let new_len = range.end - range.start;

        assert!(
            range.end <= self.len,
            "Slice range end {} exceeds buffer length {}",
            range.end,
            self.len
        );

        Arc::new(CudaDeviceBuffer {
            inner: Arc::clone(&self.inner),
            offset: new_offset,
            len: new_len,
            device_ptr: self.device_ptr,
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}
