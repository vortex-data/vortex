// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::cmp::min;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::ops::Range;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::CudaView;
use cudarc::driver::CudaViewMut;
use cudarc::driver::DevicePtr;
use cudarc::driver::DeviceRepr;
use cudarc::driver::sys;
use futures::executor::block_on;
use futures::future::BoxFuture;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBuffer;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBuffer;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex::error::vortex_panic;

use crate::stream::await_stream_callback;

/// cuDF may read Arrow validity masks through a 64-byte padded extent, so keep
/// exported masks logically sliced but zero-pad their backing allocation.
pub(crate) const CUDF_VALIDITY_BUFFER_PADDING: usize = 64;

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
    /// Allocation-relative byte offset where zeroed tail padding begins.
    ///
    /// `None` means the tail contents are not tracked. Arrow validity export uses this to decide
    /// whether a sliced logical bitmap can safely reuse this allocation for cuDF's padded mask
    /// reads without copying or repacking.
    zeroed_tail_start: Option<usize>,
}

mod private {
    use std::fmt::Debug;
    use std::sync::Arc;

    use cudarc::driver::CudaSlice;
    use cudarc::driver::CudaStream;
    use cudarc::driver::CudaView;
    use cudarc::driver::CudaViewMut;
    use cudarc::driver::DeviceRepr;
    use vortex::buffer::Alignment;
    use vortex::error::VortexExpect;

    pub trait DeviceAllocation: Debug + Send + Sync + 'static {
        /// Get the minimum alignment of the allocation.
        fn alignment(&self) -> Alignment;

        /// Get a reference to the underlying cuStream.
        fn stream(&self) -> &Arc<CudaStream>;

        /// Access the values as a bytes view.
        fn as_bytes_view(&self) -> CudaView<'_, u8>;

        /// Access the values as a mutable bytes view.
        fn as_bytes_view_mut(&mut self) -> CudaViewMut<'_, u8>;
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
            // SAFETY: all types can be reinterpreted as a byte slice.
            let result = unsafe { self.as_view().transmute::<u8>(bytes_len) };
            result.vortex_expect("Downcasting CudaSlice<T> => CudaSlice<u8> must succeed")
        }

        fn as_bytes_view_mut(&mut self) -> CudaViewMut<'_, u8> {
            let bytes_len = self.len() * size_of::<T>();
            // SAFETY: all types can be reinterpreted as a mutable byte slice.
            let result = unsafe { self.transmute_mut::<u8>(bytes_len) };
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
            zeroed_tail_start: None,
        }
    }

    /// Wraps a CUDA allocation and records that bytes from `zeroed_tail_start` are already zeroed.
    pub(crate) fn new_with_zeroed_tail<T: DeviceRepr + Debug + Send + Sync + 'static>(
        cuda_slice: CudaSlice<T>,
        zeroed_tail_start: usize,
    ) -> VortexResult<Self> {
        let mut buffer = Self::new(cuda_slice);
        vortex_ensure!(
            zeroed_tail_start <= buffer.len,
            "zeroed tail start {zeroed_tail_start} exceeds CUDA allocation length {}",
            buffer.len
        );
        buffer.zeroed_tail_start = Some(zeroed_tail_start);
        Ok(buffer)
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

        // SAFETY: All DeviceRepr types are aligned to < 256 bytes, which is what CUDA allocator
        // gives us back. So we should not suffer any alignment issues at runtime.
        unsafe {
            self.allocation
                .as_bytes_view()
                .slice(self.offset..self.offset + self.len)
                .transmute::<T>(new_len)
                .vortex_expect("Failed to transmute from CudaView<u8> to CudaView<T>")
        }
    }

    pub(crate) fn with_view_mut<T: DeviceRepr + 'static, R>(
        &mut self,
        function: impl FnOnce(&mut CudaViewMut<'_, T>) -> R,
    ) -> Option<R> {
        let new_len = self.len / size_of::<T>();
        let allocation = Arc::get_mut(&mut self.allocation)?;
        let mut bytes = allocation.as_bytes_view_mut();
        let mut bytes = bytes.slice_mut(self.offset..self.offset + self.len);
        // SAFETY: The same alignment and size requirements as `as_view` apply, and unique
        // ownership of both allocation Arcs guarantees no other device view can alias this one.
        let mut values = unsafe { bytes.transmute_mut::<T>(new_len) }
            .vortex_expect("Failed to transmute from CudaViewMut<u8> to CudaViewMut<T>");
        Some(function(&mut values))
    }
}

#[cfg(test)]
pub(crate) fn cuda_backing_allocation(handle: &BufferHandle) -> VortexResult<BufferHandle> {
    let device_buffer = handle
        .as_device_opt()
        .ok_or_else(|| vortex_err!("Buffer is not on device"))?;

    let cuda_buf = device_buffer
        .as_any()
        .downcast_ref::<CudaDeviceBuffer>()
        .ok_or_else(|| vortex_err!("expected CudaDeviceBuffer, was {device_buffer:?}"))?;
    let len = cuda_buf.allocation.as_bytes_view().len();

    Ok(BufferHandle::new_device(Arc::new(CudaDeviceBuffer {
        allocation: Arc::clone(&cuda_buf.allocation),
        offset: 0,
        len,
        device_ptr: cuda_buf.device_ptr,
        alignment: cuda_buf.alignment,
        zeroed_tail_start: cuda_buf.zeroed_tail_start,
    })))
}

/// Extension trait for getting CUDA views from a [`BufferHandle`].
pub trait CudaBufferExt {
    /// Returns a readonly [`CudaView`] for the buffer handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is not a CUDA buffer.
    fn cuda_view<T: DeviceRepr + Send + Sync + 'static>(&self) -> VortexResult<CudaView<'_, T>>;

    /// Consumes and returns this handle, calling `function` with a mutable CUDA view when the
    /// handle and backing allocation are unique.
    ///
    /// The returned option is `None` when another handle references the same device buffer or
    /// allocation.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is not a CUDA buffer.
    fn with_cuda_view_mut<T: DeviceRepr + Send + Sync + 'static, R>(
        self,
        function: impl FnOnce(&mut CudaViewMut<'_, T>) -> R,
    ) -> VortexResult<(Self, Option<R>)>
    where
        Self: Sized;

    /// Returns the on-device pointer for the start of the buffer handle.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is not a CUDA buffer.
    fn cuda_device_ptr(&self) -> VortexResult<sys::CUdeviceptr>;

    /// Returns whether this buffer has zeroed tail padding for the requested extent.
    ///
    /// # Errors
    ///
    /// Returns an error if the buffer is not a CUDA buffer.
    fn has_zeroed_tail_padding(&self, logical_len: usize, padded_len: usize) -> VortexResult<bool>;
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

    fn with_cuda_view_mut<T: DeviceRepr + Send + Sync + 'static, R>(
        self,
        function: impl FnOnce(&mut CudaViewMut<'_, T>) -> R,
    ) -> VortexResult<(Self, Option<R>)> {
        if !self.is_on_device() {
            return Err(vortex_err!("Buffer is not on device"));
        }

        let device_buffer = self.unwrap_device();
        if !device_buffer.as_any().is::<CudaDeviceBuffer>() {
            return Err(vortex_err!(
                "expected CudaDeviceBuffer, was {device_buffer:?}"
            ));
        }

        let device_buffer: Arc<dyn Any + Send + Sync> = device_buffer;
        let mut cuda_buf = Arc::downcast::<CudaDeviceBuffer>(device_buffer)
            .map_err(|_| vortex_err!("CudaDeviceBuffer downcast failed after type check"))?;
        let result = Arc::get_mut(&mut cuda_buf).and_then(|buffer| buffer.with_view_mut(function));

        Ok((BufferHandle::new_device(cuda_buf), result))
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

    fn has_zeroed_tail_padding(&self, logical_len: usize, padded_len: usize) -> VortexResult<bool> {
        let device_buffer = self
            .as_device_opt()
            .ok_or_else(|| vortex_err!("Buffer is not on device"))?;

        let cuda_buf = device_buffer
            .as_any()
            .downcast_ref::<CudaDeviceBuffer>()
            .ok_or_else(|| vortex_err!("expected CudaDeviceBuffer, was {device_buffer:?}"))?;

        if logical_len > padded_len || logical_len > cuda_buf.len {
            return Ok(false);
        }

        let Some(required_end) = cuda_buf.offset.checked_add(padded_len) else {
            return Ok(false);
        };
        if required_end > cuda_buf.allocation.as_bytes_view().len() {
            return Ok(false);
        }

        if logical_len == padded_len {
            return Ok(true);
        }

        let Some(zeroed_tail_start) = cuda_buf.zeroed_tail_start else {
            return Ok(false);
        };
        let Some(logical_end) = cuda_buf.offset.checked_add(logical_len) else {
            return Ok(false);
        };
        Ok(zeroed_tail_start <= logical_end)
    }
}

impl Debug for CudaDeviceBuffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaDeviceBuffer")
            .field("allocation", &self.allocation)
            .field("device_ptr", &self.device_ptr)
            .field("offset", &self.offset)
            .field("len", &self.len)
            .field("zeroed_tail_start", &self.zeroed_tail_start)
            .finish()
    }
}

impl Hash for CudaDeviceBuffer {
    fn hash<H: Hasher>(&self, state: &mut H) {
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
        block_on(self.copy_to_host(alignment)?)
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
        let source = self.as_view::<u8>();
        // Use cudarc's read guard so this allocation stream waits for writes submitted on any
        // other managed stream before scheduling the device-to-host copy.
        let (src_ptr, record_read) = source.device_ptr(stream);

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
        drop(record_read);

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
            zeroed_tail_start: self.zeroed_tail_start,
        })
    }

    fn as_any(&self) -> &dyn Any {
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
                zeroed_tail_start: self.zeroed_tail_start,
            }))
        } else if alignment > Alignment::new(256) {
            vortex_panic!("we do not support alignment greater than 256")
        } else {
            vortex_panic!("some how we alloc a cuda buffer with alignment less than 256")
        }
    }
}

#[cfg(test)]
mod tests {
    use cudarc::driver::CudaContext;
    use cudarc::driver::PushKernelArg;
    use vortex::buffer::Buffer;
    use vortex::dtype::PType;
    use vortex::error::VortexResult;
    use vortex::error::vortex_bail;

    use super::*;
    use crate::CudaSession;

    #[crate::test]
    async fn mutable_view_requires_unique_ownership() -> VortexResult<()> {
        let ctx = CudaSession::create_execution_ctx(&crate::cuda_session())?;
        let allocation = ctx.device_alloc::<u32>(4)?;
        let mut handle = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(allocation)));

        let (next, result) = handle.with_cuda_view_mut::<u32, _>(|view| view.len())?;
        handle = next;
        assert_eq!(result, Some(4));

        let alias = handle.clone();
        let (next, result) = handle.with_cuda_view_mut::<u32, _>(|view| view.len())?;
        handle = next;
        assert!(result.is_none());
        drop(alias);

        let (_, result) = handle.with_cuda_view_mut::<u32, _>(|view| view.len())?;
        assert_eq!(result, Some(4));
        Ok(())
    }

    #[crate::test]
    async fn copy_to_host_waits_for_write_on_another_stream() -> VortexResult<()> {
        let context = CudaContext::new(0)
            .map_err(|err| vortex_err!("failed to create CUDA context: {err}"))?;
        let cuda_session = CudaSession::with_stream_pool_capacity(context, 2);
        let session = vortex::array::array_session().with_some(cuda_session);
        let allocation_ctx = CudaSession::create_execution_ctx(&session)?;
        let mut launch_ctx = CudaSession::create_execution_ctx(&session)?;
        assert_ne!(
            allocation_ctx.stream().cu_stream(),
            launch_ctx.stream().cu_stream()
        );

        let handle = allocation_ctx.copy_to_device(vec![0u32; 4])?.await?;
        let kernel = launch_ctx.load_function("constant_numeric", &[PType::U32])?;
        let value = 42u32;
        let len = 4u64;
        let (handle, launch) = handle.with_cuda_view_mut::<u32, _>(|view| {
            launch_ctx.launch_kernel(&kernel, 4, |args| {
                args.arg(view).arg(&value).arg(&len);
            })
        })?;
        let Some(launch) = launch else {
            vortex_bail!("fresh CUDA allocation was unexpectedly shared");
        };
        launch?;

        let values = Buffer::<u32>::from_byte_buffer(handle.try_to_host()?.await?);
        assert_eq!(values.as_slice(), &[42; 4]);
        Ok(())
    }
}
