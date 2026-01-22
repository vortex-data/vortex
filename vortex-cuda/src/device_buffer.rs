// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::CudaView;
use cudarc::driver::DevicePtr;
use cudarc::driver::DeviceRepr;
use cudarc::driver::sys;
use futures::future::BoxFuture;
use vortex_array::buffer::BufferHandle;
use vortex_array::buffer::DeviceBuffer;
use vortex_buffer::Alignment;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::stream::await_stream_callback;

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
    fn copy_to_host_sync(&self, alignment: Alignment) -> VortexResult<ByteBuffer> {
        let mut host_buffer = BufferMut::<T>::with_capacity_aligned(self.len, alignment);

        // Add offset to device pointer to account for any previous slicing operations.
        let src_ptr = self.device_ptr + (self.offset * size_of::<T>()) as u64;

        // SAFETY: We pass a valid pointer to a buffer with sufficient capacity.
        // `cuMemcpyDtoHAsync_v2` fully initializes the memory.
        unsafe {
            sys::cuMemcpyDtoH_v2(
                host_buffer.spare_capacity_mut().as_mut_ptr().cast(),
                src_ptr,
                self.len * size_of::<T>(),
            )
            .result()
            .map_err(|e| vortex_err!("Failed to copy from device to host: {}", e))?;
        }

        // SAFETY: `cuMemcpyDtoHAsync_v2` fully initialized the buffer.
        unsafe {
            host_buffer.set_len(self.len);
        }

        Ok(host_buffer.freeze().into_byte_buffer())
    }

    /// Copies a device buffer to host memory asynchronously.
    ///
    /// Allocates host memory, schedules an async copy, and returns a future
    /// that completes when the copy is finished.
    ///
    /// # Arguments
    ///
    /// * `alignment` - The memory alignment to use for the host buffer.
    ///
    /// # Returns
    ///
    /// A future that resolves to the host buffer when the copy completes.
    fn copy_to_host(
        &self,
        alignment: Alignment,
    ) -> VortexResult<BoxFuture<'static, VortexResult<ByteBuffer>>> {
        let stream = self.inner.stream();

        // Add offset to device pointer to account for any previous slicing operations.
        let src_ptr = self.device_ptr + (self.offset * size_of::<T>()) as u64;

        let mut host_buffer: BufferMut<T> = BufferMut::with_capacity_aligned(self.len, alignment);
        let len = self.len;

        // SAFETY: We pass a valid pointer to a buffer with sufficient capacity.
        // `cuMemcpyDtoHAsync_v2` fully initializes the memory.
        unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                host_buffer.spare_capacity_mut().as_mut_ptr().cast(),
                src_ptr,
                len * size_of::<T>(),
                stream.cu_stream(),
            )
            .result()
            .map_err(|e| vortex_err!("Failed to schedule async copy to host: {}", e))?;
        }

        let cuda_slice = Arc::clone(&self.inner);

        Ok(Box::pin(async move {
            await_stream_callback(cuda_slice.stream()).await?;

            // Keep device memory alive until copy completes.
            let _keep_alive = cuda_slice;

            // SAFETY: `cuMemcpyDtoHAsync_v2` fully initialized the buffer.
            unsafe {
                host_buffer.set_len(len);
            }

            Ok(host_buffer.freeze().into_byte_buffer())
        }))
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
