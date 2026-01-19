// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::DeviceRepr;
use vortex_array::buffer::DeviceBuffer;
use vortex_buffer::Alignment;
use vortex_buffer::BufferMut;
use vortex_buffer::ByteBuffer;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

/// A CUDA device buffer wrapping a [`CudaSlice<T>`].
pub struct CudaDeviceBuffer<T> {
    cuda_slice: CudaSlice<T>,
}

impl<T> CudaDeviceBuffer<T> {
    /// Creates a new CUDA device buffer from a [`CudaSlice`].
    pub fn new(cuda_slice: CudaSlice<T>) -> Self {
        Self { cuda_slice }
    }

    /// Returns a reference to the underlying [`CudaSlice<T>`].
    pub fn cuda_slice(&self) -> &CudaSlice<T> {
        &self.cuda_slice
    }
}

impl<T> Debug for CudaDeviceBuffer<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDeviceBuffer")
            .field(
                "address",
                &(&raw const self.cuda_slice as *const _ as usize),
            )
            .field("num_bytes", &self.cuda_slice.num_bytes())
            .finish()
    }
}

impl<T: 'static> Hash for CudaDeviceBuffer<T> {
    /// Hash the buffer pointer address.
    fn hash<H: Hasher>(&self, state: &mut H) {
        (&raw const self.cuda_slice).hash(state);
    }
}

impl<T: 'static> PartialEq for CudaDeviceBuffer<T> {
    /// Compares two buffers by pointer address.
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(&raw const self.cuda_slice, &raw const other.cuda_slice)
    }
}

impl<T: DeviceRepr + Clone + Send + Sync + 'static> DeviceBuffer for CudaDeviceBuffer<T> {
    /// Returns the number of elements in the CUDA device buffer of type T.
    fn len(&self) -> usize {
        self.cuda_slice.len()
    }

    /// Copies the CUDA device buffer to host memory.
    ///
    /// Allocates a host buffer with the specified alignment and copies the data
    /// from the device to the host. The operation is implicitly synchronized
    /// when the underlying event is dropped.
    ///
    /// # Arguments
    ///
    /// * `alignment` - The byte alignment for the allocated host buffer.
    ///
    /// # Returns
    ///
    /// A `ByteBuffer` containing the copied data, or an error if the copy fails.
    fn copy_to_host(&self, alignment: Alignment) -> VortexResult<ByteBuffer> {
        let len = self.cuda_slice.len();
        let mut host_buffer = BufferMut::<T>::with_capacity_aligned(len, alignment);

        // TODO(0ax1): Make the memcopy to host async. Even though `memcpy_dtoh`
        // uses into `memcpy_dtoh_async`, it implicitly calls synchronize on the
        // stream when dropping the `SyncOnDrop` `_record_dst` event at the end
        // of the function.
        self.cuda_slice
            .stream()
            .memcpy_dtoh(&self.cuda_slice, unsafe {
                // SAFETY: We allocated sufficient capacity and fill the entire buffer.
                host_buffer.set_len(len);
                host_buffer.as_mut_slice()
            })
            .map_err(|e| vortex_err!("Failed to copy from device to host: {}", e))?;

        Ok(host_buffer.freeze().into_byte_buffer())
    }

    /// Slices the CUDA device buffer to a subrange.
    fn slice(&self, _range: Range<usize>) -> Arc<dyn DeviceBuffer> {
        // TODO(0ax1): impl slice on CUDA slice
        unimplemented!("CudaDeviceBuffer::slice is not yet implemented")
    }
}
