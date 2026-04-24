// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::cmp::min;
use std::fmt::Debug;
use std::ops::Range;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::CudaView;
use cudarc::driver::DevicePtr;
use cudarc::driver::DeviceRepr;
use cudarc::driver::sys;
use futures::future::BoxFuture;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBuffer;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBuffer;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;
use vortex::error::vortex_panic;

use crate::stream::await_stream_callback;

/// A [`DeviceBuffer`] wrapping a CUDA GPU allocation.
///
/// Like the host `BufferHandle` variant, all slicing/referencing works in terms of byte units.
#[derive(Clone)]
pub struct CudaDeviceBuffer {
    allocation: Arc<dyn private::DeviceAllocation>,
    /// Offset in bytes from the start of the allocation
    offset: usize,
    /// Length in bytes
    len: usize,
    /// CUDA device pointer
    device_ptr: u64,
    /// Minimum required alignment of the buffer.
    alignment: Alignment,
}

mod private {
    use std::fmt::Debug;
    use std::sync::Arc;

    use cudarc::driver::CudaSlice;
    use cudarc::driver::CudaStream;
    use cudarc::driver::CudaView;
    use cudarc::driver::DeviceRepr;
    use vortex::buffer::Alignment;
    use vortex::error::VortexExpect;

    pub trait DeviceAllocation: Debug + Send + Sync + 'static {
        /// Get the minimum alignment of the allocation.
        fn alignment(&self) -> Alignment;

        /// Get a reference to the underlying cuStream.
        fn stream(&self) -> &Arc<CudaStream>;

        /// Access the values as a bytes view
        fn as_bytes_view(&self) -> CudaView<'_, u8>;
    }

    // CudaSlice needs to be held by the CudaDeviceBuffer
    impl<T: DeviceRepr + Debug + Send + Sync + 'static> DeviceAllocation for CudaSlice<T> {
        fn alignment(&self) -> Alignment {
            Alignment::of::<T>()
        }

        fn stream(&self) -> &Arc<CudaStream> {
            self.stream()
        }

        fn as_bytes_view(&self) -> CudaView<'_, u8> {
            let bytes_len = self.len() * size_of::<T>();
            // SAFETY: all types can be reinterpreted as a byte slice
            let result = unsafe { self.as_view().transmute::<u8>(bytes_len) };
            result.vortex_expect("Downcasting CudaSlice<T> => CudaSlice<u8> must succeed")
        }
    }
}

impl CudaDeviceBuffer {
    /// Creates a new CUDA device buffer from a [`CudaSlice<T>`].
    ///
    /// The device buffer itself is type-erased and only works in terms of bytes, similar to the
    /// `BufferHandle` interface that it implements.
    pub fn new<T: DeviceRepr + Debug + Send + Sync + 'static>(cuda_slice: CudaSlice<T>) -> Self {
        let len = cuda_slice.len() * size_of::<T>();
        let (device_ptr, _) = cuda_slice.device_ptr(cuda_slice.stream());

        Self {
            allocation: Arc::new(cuda_slice),
            offset: 0,
            len,
            device_ptr,
            alignment: Alignment::of::<T>(),
        }
    }

    /// Returns the byte offset within the allocated buffer.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// Returns the adjusted device pointer accounting for the offset.
    pub fn offset_ptr(&self) -> sys::CUdeviceptr {
        self.device_ptr + self.offset as u64
    }

    /// Returns a [`CudaView`] to the CUDA device buffer.
    pub fn as_view<T: DeviceRepr + 'static>(&self) -> CudaView<'_, T> {
        // Return a new &[T]
        let new_len = self.len / size_of::<T>();

        // SAFETY: All DeviecRepr types are aligned to < 256 bytes, which is what CUDA allocator
        //  gives us back. So we should not suffer any alignment issues at runtime.
        unsafe {
            self.allocation
                .as_bytes_view()
                .slice(self.offset..self.offset + self.len)
                .transmute::<T>(new_len)
                .vortex_expect("Failed to transmute from CudaView<u8> to CudaView<T>")
        }
    }
}

// TODO(aduffy): we should add cuda_view_mut and enforce the borrow rules. This is a bit tricky
//  because many executions are async, we should lean into that with ownership and having any
//  async context actions take ownership of the buffer and return ownership when they're done.
/// Extension trait for getting a [`CudaView`] from a [`BufferHandle`].
pub trait CudaBufferExt {
    /// Returns a readonly [`CudaView`] for the buffer handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is not a CUDA buffer.
    fn cuda_view<T: DeviceRepr + Send + Sync + 'static>(&self) -> VortexResult<CudaView<'_, T>>;

    /// Returns the on-device pointer for the start of the buffer handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is not a CUDA buffer.
    fn cuda_device_ptr(&self) -> VortexResult<sys::CUdeviceptr>;
}

impl CudaBufferExt for BufferHandle {
    fn cuda_view<T: DeviceRepr + Send + Sync + 'static>(&self) -> VortexResult<CudaView<'_, T>> {
        let device_buffer = self
            .as_device_opt()
            .ok_or_else(|| vortex_err!("Buffer is not on device"))?;

        let cuda_buf = device_buffer
            .as_any()
            .downcast_ref::<CudaDeviceBuffer>()
            .ok_or_else(|| vortex_err!("expected CudaDeviceBuffer, was {device_buffer:?}"))?;

        Ok(cuda_buf.as_view::<T>())
    }

    fn cuda_device_ptr(&self) -> VortexResult<sys::CUdeviceptr> {
        let ptr = self
            .as_device_opt()
            .ok_or_else(|| vortex_err!("Buffer is not on device"))?
            .as_any()
            .downcast_ref::<CudaDeviceBuffer>()
            .ok_or_else(|| vortex_err!("expected CudaDeviceBuffer"))?
            .offset_ptr();

        Ok(ptr)
    }
}

impl Debug for CudaDeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDeviceBuffer")
            .field("allocation", &self.allocation)
            .field("device_ptr", &self.device_ptr)
            .field("offset", &self.offset)
            .field("len", &self.len)
            .finish()
    }
}

impl std::hash::Hash for CudaDeviceBuffer {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.device_ptr.hash(state);
        self.len.hash(state);
        self.offset.hash(state);
    }
}

// CUDA device buffers are equal if they point to the same extent of GPU memory
impl PartialEq for CudaDeviceBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.device_ptr == other.device_ptr && self.len == other.len && self.offset == other.offset
    }
}

impl DeviceBuffer for CudaDeviceBuffer {
    /// Returns the number of bytes in the device buffer.
    fn len(&self) -> usize {
        self.len
    }

    fn alignment(&self) -> Alignment {
        self.alignment
    }

    /// Synchronous copy of CUDA device to host memory.
    ///
    /// The copy is not started before other operations on the streams are completed.
    /// This is synonymous to doing a synchronize on the stream before the copy.
    ///
    /// The asynchronous `copy_to_host` function should be preferred whenever possible.
    ///
    /// # Arguments
    ///
    /// * `alignment` - The memory alignment to use for the host buffer.
    ///
    /// # Errors
    ///
    /// Returns an error if the CUDA memory copy operation fails.
    fn copy_to_host_sync(&self, alignment: Alignment) -> VortexResult<ByteBuffer> {
        futures::executor::block_on(self.copy_to_host(alignment)?)
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
        let stream = self.allocation.stream();

        // Add offset to device pointer to account for any previous slicing operations.
        let src_ptr = self.device_ptr + self.offset as u64;

        let mut host_buffer: ByteBufferMut =
            ByteBufferMut::with_capacity_aligned(self.len, alignment);
        let len = self.len;

        stream
            .context()
            .bind_to_thread()
            .map_err(|e| vortex_err!("Failed to bind CUDA context: {}", e))?;

        // SAFETY: We pass a valid pointer to a buffer with sufficient capacity.
        // `cuMemcpyDtoHAsync_v2` fully initializes the memory.
        unsafe {
            sys::cuMemcpyDtoHAsync_v2(
                host_buffer.spare_capacity_mut().as_mut_ptr().cast(),
                src_ptr,
                len,
                stream.cu_stream(),
            )
            .result()
            .map_err(|e| vortex_err!("Failed to schedule async copy to host: {}", e))?;
        }

        let cuda_slice = Arc::clone(&self.allocation);

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
    ///
    /// This is a byte range, not elements range, due to the DeviceBuffer interface.
    fn slice(&self, range: Range<usize>) -> Arc<dyn DeviceBuffer> {
        assert!(
            range.end <= self.len,
            "Slice range end {} exceeds allocation size {}",
            range.end,
            self.len
        );

        let new_offset = self.offset + range.start;
        let new_len = range.end - range.start;

        let trailing = (self.device_ptr + new_offset as u64).trailing_zeros();
        let exponent =
            u8::try_from(min(15, trailing)).vortex_expect("min(15, x) always fits in u8");
        let slice_align = Alignment::from_exponent(exponent);

        assert!(
            slice_align.is_aligned_to(self.allocation.alignment()),
            "slice must respect minimum alignment {}, min {}",
            slice_align,
            self.allocation.alignment()
        );

        Arc::new(CudaDeviceBuffer {
            allocation: Arc::clone(&self.allocation),
            offset: new_offset,
            len: new_len,
            device_ptr: self.device_ptr,
            alignment: self.alignment,
        })
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn aligned(self: Arc<Self>, alignment: Alignment) -> VortexResult<Arc<dyn DeviceBuffer>> {
        let effective_ptr = self.device_ptr + self.offset as u64;
        if effective_ptr.is_multiple_of(*alignment as u64) {
            Ok(Arc::new(CudaDeviceBuffer {
                allocation: Arc::clone(&self.allocation),
                offset: self.offset,
                len: self.len,
                device_ptr: self.device_ptr,
                alignment,
            }))
        } else if alignment > Alignment::new(256) {
            vortex_panic!("we do not support alignment greater than 256")
        } else {
            vortex_panic!("some how we alloc a cuda buffer with alignment less than 256")
        }
    }
}
