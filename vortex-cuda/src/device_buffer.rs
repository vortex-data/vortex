// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::any::Any;
use std::cmp::min;
use std::fmt::Debug;
use std::fmt::Formatter;
use std::hash::Hash;
use std::hash::Hasher;
use std::mem;
use std::ops::Range;
use std::sync::Arc;

use futures::executor::block_on;
use futures::future::BoxFuture;
use tracing::error;
use vortex::array::buffer::BufferHandle;
use vortex::array::buffer::DeviceBuffer;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBuffer;
use vortex::buffer::ByteBufferMut;
use vortex::error::VortexError;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;
use vortex_cuda_driver::CudaSlice;
use vortex_cuda_driver::CudaView;
use vortex_cuda_driver::CudaViewMut;
use vortex_cuda_driver::DevicePtr;
use vortex_cuda_driver::DeviceRepr;
use vortex_cuda_driver::DriverError;
use vortex_cuda_driver::sys;

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

    use vortex::buffer::Alignment;
    use vortex::error::VortexExpect;
    use vortex_cuda_driver::CudaSlice;
    use vortex_cuda_driver::CudaStream;
    use vortex_cuda_driver::CudaView;
    use vortex_cuda_driver::CudaViewMut;
    use vortex_cuda_driver::DeviceRepr;

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
    ///
    /// Returns an error if accessing or tracking the allocation's device pointer fails.
    pub fn new<T: DeviceRepr + Debug + Send + Sync + 'static>(
        cuda_slice: CudaSlice<T>,
    ) -> VortexResult<Self> {
        let len = cuda_slice.len() * size_of::<T>();
        // Keep the tracked read to order the allocation stream after writes on other streams.
        // Only the address is stored here, so finish the guard immediately; queued waits remain.
        let (device_ptr, record_read) =
            cuda_slice.device_ptr(cuda_slice.stream()).map_err(|err| {
                VortexError::external(err).with_context("Failed to access CUDA buffer pointer")
            })?;
        record_read.finish().map_err(|err| {
            VortexError::external(err).with_context("Failed to track CUDA buffer pointer access")
        })?;

        Ok(Self {
            allocation: Arc::new(cuda_slice),
            offset: 0,
            len,
            device_ptr,
            alignment: Alignment::of::<T>(),
            zeroed_tail_start: None,
        })
    }

    /// Wraps a CUDA allocation and records that bytes from `zeroed_tail_start` are already zeroed.
    pub(crate) fn new_with_zeroed_tail<T: DeviceRepr + Debug + Send + Sync + 'static>(
        cuda_slice: CudaSlice<T>,
        zeroed_tail_start: usize,
    ) -> VortexResult<Self> {
        let mut buffer = Self::new(cuda_slice)?;
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

/// Include the slice's leading bytes up to an 8-byte boundary, without extending its tail.
/// CUDA allocations are aligned, so the prefix remains within the backing allocation.
pub(crate) fn cuda_aligned_bitmap_view(
    handle: &BufferHandle,
) -> VortexResult<(CudaView<'_, u8>, usize)> {
    let device_buffer = handle
        .as_device_opt()
        .ok_or_else(|| vortex_err!("Buffer is not on device"))?;
    let cuda_buf = device_buffer
        .as_any()
        .downcast_ref::<CudaDeviceBuffer>()
        .ok_or_else(|| vortex_err!("expected CudaDeviceBuffer, was {device_buffer:?}"))?;
    let prefix_bytes = cuda_buf.offset % size_of::<u64>();
    let view = cuda_buf
        .allocation
        .as_bytes_view()
        .slice(cuda_buf.offset - prefix_bytes..cuda_buf.offset + cuda_buf.len);
    Ok((view, prefix_bytes))
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

pub(crate) fn with_cuda_view_mut<T: DeviceRepr + Send + Sync + 'static, R>(
    handle: BufferHandle,
    function: impl FnOnce(&mut CudaViewMut<'_, T>) -> R,
) -> VortexResult<(BufferHandle, Option<R>)> {
    if !handle.is_on_device() {
        return Err(vortex_err!("Buffer is not on device"));
    }

    let device_buffer = handle.unwrap_device();
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

/// Extension trait for getting CUDA views from a [`BufferHandle`].
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

/// Retains both sides of an enqueued D2H copy until completion, including an unpolled future.
struct PendingHostCopy {
    buffers: Option<(ByteBufferMut, Arc<dyn private::DeviceAllocation>)>,
    // Store this before creating the future so Drop reports it even if the future is never
    // polled. Taking it when returning the error prevents logging a returned primary error.
    tracking_error: Option<DriverError>,
}

impl PendingHostCopy {
    async fn finish(mut self, len: usize) -> VortexResult<ByteBuffer> {
        let (_, allocation) = self.buffers.as_ref().vortex_expect("copy is pending");
        await_stream_callback(allocation.stream()).await?;

        let (mut host_buffer, _keep_alive) = self.buffers.take().vortex_expect("copy is pending");
        if let Some(err) = self.tracking_error.take() {
            return Err(
                VortexError::external(err).with_context("Failed to track CUDA copy source read")
            );
        }

        // SAFETY: The successful callback confirms `cuMemcpyDtoHAsync_v2` fully initialized
        // the buffer. Both allocations have remained alive until this point.
        unsafe {
            host_buffer.set_len(len);
        }

        Ok(host_buffer.freeze().into_byte_buffer())
    }
}

impl Drop for PendingHostCopy {
    fn drop(&mut self) {
        if let Some(err) = self.tracking_error.take() {
            error!(error = ?err, "Unreturned error tracking CUDA copy source read");
        }
        let Some((host_buffer, allocation)) = self.buffers.take() else {
            return;
        };
        // Callback registration/error and future cancellation cannot prove completion. This
        // synchronization must run even when access tracking returned an error.
        if let Err(err) = allocation.stream().synchronize() {
            error!(error = ?err, "Failed to wait for CUDA host copy; retaining in-flight buffers");
            #[allow(
                clippy::mem_forget,
                reason = "without confirmed DMA completion, freeing either buffer risks use-after-free"
            )]
            mem::forget((host_buffer, allocation));
        }
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
    /// # Cancellation
    ///
    /// Dropping an unfinished future synchronizes the stream, potentially blocking
    /// on unrelated work. If synchronization fails, the error is logged and buffers
    /// and their stream/context ownership are leaked to avoid use-after-free.
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
        // Use the read guard so this allocation stream waits for writes submitted on any
        // other stream before scheduling the device-to-host copy.
        let (src_ptr, record_read) = source.device_ptr(stream).map_err(|err| {
            VortexError::external(err).with_context("Failed to access CUDA copy source")
        })?;

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
        let mut pending_copy = PendingHostCopy {
            buffers: Some((host_buffer, Arc::clone(&self.allocation))),
            tracking_error: None,
        };
        // Tracking can fail after the copy was enqueued. Do not return that error until the
        // callback has completed, while retaining both allocations even on cancellation.
        pending_copy.tracking_error = record_read.finish().err();

        Ok(Box::pin(pending_copy.finish(len)))
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
            alignment: min(self.alignment, slice_align),
            zeroed_tail_start: self.zeroed_tail_start,
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn aligned(self: Arc<Self>, alignment: Alignment) -> VortexResult<Arc<dyn DeviceBuffer>> {
        let effective_ptr = self.device_ptr + self.offset as u64;
        vortex_ensure!(
            effective_ptr.is_multiple_of(alignment.as_usize() as u64),
            "CUDA buffer cannot satisfy alignment {alignment} without copying"
        );
        Ok(Arc::new(CudaDeviceBuffer {
            allocation: Arc::clone(&self.allocation),
            offset: self.offset,
            len: self.len,
            device_ptr: self.device_ptr,
            alignment,
            zeroed_tail_start: self.zeroed_tail_start,
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::error::Error;
    use std::ffi::c_void;
    use std::sync::mpsc::Receiver;
    use std::sync::mpsc::RecvTimeoutError;
    use std::sync::mpsc::Sender;
    use std::sync::mpsc::channel;
    use std::task::Context;
    use std::task::Waker;
    use std::thread;
    use std::time::Duration;

    use rstest::rstest;
    use vortex::buffer::Buffer;
    use vortex::dtype::PType;
    use vortex::error::VortexResult;
    use vortex::error::vortex_bail;
    use vortex::session::VortexSession;
    use vortex_cuda_driver::CudaContext;
    use vortex_cuda_driver::CudaStream;
    use vortex_cuda_driver::PushKernelArg;
    use vortex_cuda_driver::stream;

    use super::*;
    use crate::CudaSession;

    #[crate::test]
    async fn alignment_update_is_metadata_only() -> VortexResult<()> {
        let ctx = CudaSession::create_execution_ctx(&crate::cuda_session())?;
        let original = ctx.copy_to_device(vec![7u8; 17])?.await?;
        let aligned = original.clone().ensure_aligned(Alignment::new(16))?;
        let original = original
            .as_device_opt()
            .expect("device buffer")
            .as_any()
            .downcast_ref::<CudaDeviceBuffer>()
            .expect("CUDA buffer");
        let aligned = aligned
            .as_device_opt()
            .expect("device buffer")
            .as_any()
            .downcast_ref::<CudaDeviceBuffer>()
            .expect("CUDA buffer");

        assert!(Arc::ptr_eq(&original.allocation, &aligned.allocation));
        assert_eq!(aligned.offset_ptr(), original.offset_ptr());
        assert_eq!(aligned.len(), original.len());
        assert_eq!(aligned.alignment(), Alignment::new(16));
        assert_eq!(aligned.zeroed_tail_start, Some(17));

        let misaligned = aligned.slice(1..17);
        for alignment in [Alignment::new(16), Alignment::new(512)] {
            let error = Arc::clone(&misaligned).aligned(alignment).unwrap_err();
            assert!(error.to_string().contains("without copying"), "{error}");
        }
        Ok(())
    }

    #[rstest]
    #[case::byte(1, Alignment::none())]
    #[case::two_bytes(2, Alignment::new(2))]
    #[case::eight_bytes(8, Alignment::new(8))]
    #[case::aligned(16, Alignment::new(16))]
    #[crate::test]
    async fn slice_reduces_alignment(
        #[case] offset: usize,
        #[case] alignment: Alignment,
    ) -> VortexResult<()> {
        let ctx = CudaSession::create_execution_ctx(&crate::cuda_session())?;
        let buffer = ctx
            .copy_to_device(vec![0u8; 32])?
            .await?
            .ensure_aligned(Alignment::new(16))?;
        let slice = buffer.slice(offset..32);
        assert_eq!(slice.alignment(), alignment);
        assert_eq!(
            slice.cuda_device_ptr()?,
            buffer.cuda_device_ptr()? + offset as u64
        );
        Ok(())
    }

    #[crate::test]
    async fn mutable_view_requires_unique_ownership() -> VortexResult<()> {
        let ctx = CudaSession::create_execution_ctx(&crate::cuda_session())?;
        let allocation = ctx.device_alloc::<u32>(4)?;
        let mut handle = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(allocation)?));

        let (next, result) = with_cuda_view_mut::<u32, _>(handle, |view| view.len())?;
        handle = next;
        assert_eq!(result, Some(4));

        let alias = handle.clone();
        let (next, result) = with_cuda_view_mut::<u32, _>(handle, |view| view.len())?;
        handle = next;
        assert!(result.is_none());
        drop(alias);

        let (_, result) = with_cuda_view_mut::<u32, _>(handle, |view| view.len())?;
        assert_eq!(result, Some(4));
        Ok(())
    }

    #[crate::test]
    async fn host_copy_completion_returns_tracking_error() -> VortexResult<()> {
        let ctx = CudaSession::create_execution_ctx(&crate::cuda_session())?;
        let buffer = CudaDeviceBuffer::new(ctx.device_alloc::<u8>(1)?)?;
        // Inject only the tracking result, not a GPU fault. A zero-byte completion still
        // takes the real callback path before returning the original driver error.
        let tracking_error = DriverError(sys::CUresult::CUDA_ERROR_INVALID_VALUE);
        let pending_copy = PendingHostCopy {
            buffers: Some((
                ByteBufferMut::with_capacity(0),
                Arc::clone(&buffer.allocation),
            )),
            tracking_error: Some(tracking_error),
        };
        let error = pending_copy.finish(0).await.unwrap_err();
        assert_eq!(
            error
                .source()
                .and_then(|source| source.downcast_ref::<DriverError>()),
            Some(&tracking_error)
        );
        assert!(
            error
                .to_string()
                .contains("Failed to track CUDA copy source read")
        );
        assert_eq!(Arc::strong_count(&buffer.allocation), 1);
        Ok(())
    }

    fn block_stream(stream: &CudaStream) -> VortexResult<Sender<()>> {
        let (release, receiver) = channel::<()>();
        stream
            .context()
            .bind_to_thread()
            .map_err(VortexError::external)?;
        unsafe extern "C" fn wait(data: *mut c_void) {
            // SAFETY: Successful registration transfers sole ownership of the receiver here.
            let receiver = unsafe { Box::from_raw(data.cast::<Receiver<()>>()) };
            // Sender drop releases the gate on unwind; the timeout also bounds cleanup if
            // a CUDA call synchronizes before Rust can unwind.
            let _ = receiver.recv_timeout(Duration::from_secs(5));
        }
        let data = Box::into_raw(Box::new(receiver));
        // SAFETY: The stream is live, and the callback owns data only on successful registration.
        unsafe {
            stream::launch_host_function(stream.cu_stream(), wait, data.cast()).map_err(
                |error| {
                    drop(Box::from_raw(data));
                    VortexError::external(error)
                },
            )?;
        }
        Ok(release)
    }

    #[rstest]
    #[case::unpolled(false)]
    #[case::pending(true)]
    #[crate::test]
    fn host_copy_cancellation_waits_before_releasing_allocation(
        #[case] poll: bool,
    ) -> VortexResult<()> {
        // Keep the gated stream independent of tests synchronizing the primary context.
        let context = CudaContext::new_non_primary(0, 0).map_err(VortexError::external)?;
        let session = VortexSession::empty().with_some(CudaSession::with_context(context));
        let ctx = CudaSession::create_execution_ctx(&session)?;
        let mut allocation = ctx.device_alloc::<u8>(4096)?;
        ctx.stream()
            .memset_zeros(&mut allocation)
            .map_err(VortexError::external)?;
        let buffer = CudaDeviceBuffer::new(allocation)?;
        let allocation = Arc::downgrade(&buffer.allocation);
        let mut copy = buffer.copy_to_host(Alignment::of::<u8>())?;
        // Complete the DMA before gating: a broken cancellation path must fail assertions,
        // not write into freed host memory. The future still owns both allocations.
        ctx.stream().synchronize().map_err(VortexError::external)?;
        drop(buffer);
        assert_eq!(allocation.strong_count(), 1);

        thread::scope(|scope| -> VortexResult<()> {
            // Release before the future is dropped or the scope joins its worker on unwind.
            let gate = block_stream(ctx.stream())?;
            if poll {
                assert!(
                    copy.as_mut()
                        .poll(&mut Context::from_waker(Waker::noop()))
                        .is_pending()
                );
            }
            let (started, starting) = channel();
            let (finished, completion) = channel();
            scope.spawn(move || {
                started.send(()).unwrap();
                drop(copy);
                let _ = finished.send(());
            });
            starting.recv_timeout(Duration::from_secs(5)).unwrap();
            assert_eq!(
                completion.recv_timeout(Duration::from_millis(250)),
                Err(RecvTimeoutError::Timeout),
                "cancellation returned while stream work was outstanding"
            );
            // A CudaSlice destructor could itself wait, so checking drop completion alone
            // is insufficient: PendingHostCopy must retain the allocation while waiting.
            assert_eq!(allocation.strong_count(), 1);
            drop(gate);
            completion.recv_timeout(Duration::from_secs(5)).unwrap();
            Ok(())
        })?;
        assert!(allocation.upgrade().is_none());
        Ok(())
    }

    #[crate::test]
    async fn copy_to_host_waits_for_write_on_another_stream() -> VortexResult<()> {
        let context = CudaContext::new(0)
            .map_err(|err| vortex_err!("failed to create CUDA context: {err}"))?;
        let cuda_session = CudaSession::with_context(context).with_stream_pool_capacity(2);
        let session = VortexSession::empty().with_some(cuda_session);
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
        let (handle, launch) = with_cuda_view_mut::<u32, _>(handle, |view| {
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
