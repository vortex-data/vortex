// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA stream utility functions.

use std::fmt::Debug;
use std::mem::size_of;
use std::mem::size_of_val;
use std::ops::Deref;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DeviceRepr;
use cudarc::driver::ValidAsZeroBits;
use cudarc::driver::result::stream;
use futures::future::BoxFuture;
use kanal::Sender;
use tracing::warn;
use vortex::array::buffer::BufferHandle;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;

use crate::CudaDeviceBuffer;
use crate::device_buffer::CUDF_VALIDITY_BUFFER_PADDING;

// cuDF imports Arrow validity masks into padded buffers and kernels may read through that
// padded extent. Keep copied device buffers padded and zero-tailed so Arrow validity exports
// can safely reuse matching bitmaps without repacking.

#[derive(Clone)]
pub struct VortexCudaStream(pub(crate) Arc<CudaStream>);

impl Deref for VortexCudaStream {
    type Target = Arc<CudaStream>;

    fn deref(&self) -> &Arc<CudaStream> {
        &self.0
    }
}

impl VortexCudaStream {
    /// Allocates a typed buffer on the GPU.
    ///
    /// Note: Allocation is async in case the CUDA driver supports this.
    ///
    /// The condition for alloc to be async is support for memory pools:
    /// `CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED`.
    ///
    /// Any kernel submitted to the stream after alloc can safely use the
    /// memory, as operations on the stream are ordered sequentially.
    pub(crate) fn device_alloc<T: DeviceRepr + Send + Sync + 'static>(
        &self,
        len: usize,
    ) -> VortexResult<CudaSlice<T>> {
        // SAFETY: No safety guarantees for allocations on the GPU.
        unsafe {
            self.alloc::<T>(len)
                .map_err(|e| vortex_err!("Failed to allocate device memory: {}", e))
        }
    }

    /// Copies host data to the device.
    ///
    /// Allocates device memory, schedules an async copy, and returns a future
    /// that completes when the copy is finished. The source data is moved into
    /// the future to ensure it remains valid until the copy completes.
    ///
    /// For **pageable** host memory, `memcpy_htod_async` stages the source
    /// synchronously before returning. For **pinned** host memory the transfer
    /// is truly async and the source must stay alive until the copy completes
    /// (guaranteed by the returned future capturing it).
    ///
    /// The returned [`BufferHandle`] keeps the source byte length, while its
    /// CUDA allocation may include zeroed tail padding for consumers such as cuDF
    /// that read validity masks through padded extents.
    pub(crate) fn copy_to_device<T, D>(
        &self,
        data: D,
    ) -> VortexResult<BoxFuture<'static, VortexResult<BufferHandle>>>
    where
        T: DeviceRepr + ValidAsZeroBits + Debug + Send + Sync + 'static,
        D: AsRef<[T]> + Send + 'static,
    {
        let host_slice: &[T] = data.as_ref();
        let byte_count = size_of_val(host_slice);
        let allocation_len = padded_device_allocation_len::<T>(byte_count)?;
        // `device_alloc` binds the CUDA context to the current thread.
        let mut cuda_slice: CudaSlice<T> = self.device_alloc::<T>(allocation_len)?;

        let mut values = cuda_slice.slice_mut(..host_slice.len());
        self.memcpy_htod(host_slice, &mut values)
            .map_err(|e| vortex_err!("Failed to schedule H2D copy: {}", e))?;

        zero_padding(self, &mut cuda_slice, host_slice.len())?;

        // `zero_padding` zeroed all allocation bytes after `byte_count`.
        let cuda_buf = CudaDeviceBuffer::new_with_zeroed_tail(cuda_slice, byte_count)?;
        let buffer = BufferHandle::new_device(Arc::new(cuda_buf)).slice(0..byte_count);
        let stream = Arc::clone(&self.0);

        Ok(Box::pin(async move {
            await_stream_callback(&stream).await?;

            // Keep source memory alive until copy completes.
            let _keep_alive = data;

            Ok(buffer)
        }))
    }

    /// Synchronous variant of [`copy_to_device`](Self::copy_to_device).
    ///
    /// Allocates device memory, enqueues the H2D copy on the stream, and
    /// returns immediately. The device pointer is valid as soon as this call
    /// returns; the copy completes before any later work on the same stream.
    ///
    /// For **pageable** host memory (the common case), `memcpy_htod` stages
    /// the source into a driver-managed pinned buffer before returning, so
    /// the source data is safe to drop after this call.
    ///
    /// Like [`copy_to_device`](Self::copy_to_device), this preserves the source
    /// byte length on the returned handle while keeping any tail padding in the
    /// backing CUDA allocation.
    pub(crate) fn copy_to_device_sync<T>(&self, data: &[T]) -> VortexResult<BufferHandle>
    where
        T: DeviceRepr + ValidAsZeroBits + Debug + Send + Sync + 'static,
    {
        let byte_count = size_of_val(data);
        let allocation_len = padded_device_allocation_len::<T>(byte_count)?;
        let mut cuda_slice: CudaSlice<T> = self.device_alloc(allocation_len)?;

        let mut values = cuda_slice.slice_mut(..data.len());
        self.memcpy_htod(data, &mut values)
            .map_err(|e| vortex_err!("Failed to schedule H2D copy: {}", e))?;

        zero_padding(self, &mut cuda_slice, data.len())?;

        // `zero_padding` zeroed all allocation bytes after `byte_count`.
        let cuda_buf = CudaDeviceBuffer::new_with_zeroed_tail(cuda_slice, byte_count)?;
        Ok(BufferHandle::new_device(Arc::new(cuda_buf)).slice(0..byte_count))
    }
}

/// Returns the typed CUDA allocation length for `byte_count`.
///
/// The backing allocation is padded for consumers such as cuDF that read validity masks
/// through padded extents. The returned length is in `T` elements.
fn padded_device_allocation_len<T>(byte_count: usize) -> VortexResult<usize> {
    let element_size = size_of::<T>();
    vortex_ensure!(
        element_size != 0,
        "cannot copy zero-sized values to CUDA device"
    );
    let min_allocation_bytes = byte_count.next_multiple_of(CUDF_VALIDITY_BUFFER_PADDING);
    Ok(min_allocation_bytes.div_ceil(element_size))
}

/// Zeroes the allocation tail after the copied values.
///
/// Returned handles are sliced to the copied byte count; the trailing padding
/// exists so padded mask reads stay within the backing allocation.
fn zero_padding<T: DeviceRepr + ValidAsZeroBits>(
    stream: &VortexCudaStream,
    cuda_slice: &mut CudaSlice<T>,
    copied_len: usize,
) -> VortexResult<()> {
    if copied_len >= cuda_slice.len() {
        return Ok(());
    }

    let mut padding = cuda_slice.slice_mut(copied_len..);
    stream
        .memset_zeros(&mut padding)
        .map_err(|e| vortex_err!("Failed to zero device buffer padding: {}", e))
}

/// Registers a callback and asynchronously waits for its completion.
///
/// This function can be used to asynchronously wait for events previously
/// submitted to the stream to complete, e.g. async buffer allocations.
///
/// Note: This is not equivalent to calling sync on a stream but only awaits
/// the registered callback to complete.
///
/// # Arguments
///
/// * `stream` - The CUDA stream to wait on
///
/// # Errors
///
/// Returns an error if registering the stream callback fails or if the callback
/// channel closes unexpectedly.
pub(crate) async fn await_stream_callback(stream: &CudaStream) -> VortexResult<()> {
    let rx = register_stream_callback(stream)?;

    rx.recv()
        .await
        .map_err(|e| vortex_err!("CUDA stream callback channel closed unexpectedly: {}", e))
}

/// Registers a host function callback on the stream.
///
/// # Returns
///
/// An async receiver that receives a message when all preceding work on the
/// stream completes.
///
/// # Errors
///
/// Returns an error if registering the host callback function fails.
fn register_stream_callback(stream: &CudaStream) -> VortexResult<kanal::AsyncReceiver<()>> {
    let (tx, rx) = kanal::bounded::<()>(1);

    let tx_ptr = Box::into_raw(Box::new(tx));

    stream
        .context()
        .bind_to_thread()
        .map_err(|e| vortex_err!("Failed to bind CUDA context: {}", e))?;

    /// Called from CUDA driver thread when all preceding work on the stream completes.
    unsafe extern "C" fn callback(user_data: *mut std::ffi::c_void) {
        // SAFETY: The memory of `tx` is manually managed has not been freed
        // before. We have unique ownership and can therefore free it.
        let tx = unsafe { Box::from_raw(user_data as *mut Sender<()>) };

        // Blocking send as we're in a callback invoked by the CUDA driver.
        // NOTE: send can fail if the CudaEvent is dropped by the caller, in which case the receiver
        //  is closed and sends will fail.
        if let Err(_e) = tx.send(()) {
            warn!(error = ?_e, "register_stream_callback send failed due to error");
        }
    }

    // SAFETY:
    // 1. Valid handle from the borrowed `CudaStream`.
    // 2. Valid function pointer with the the correct signature
    // 3. Valid user data pointer which is consumed exactly once
    unsafe {
        stream::launch_host_function(
            stream.cu_stream(),
            callback,
            tx_ptr as *mut std::ffi::c_void,
        )
        .map_err(|err| {
            // SAFETY: Registration failed, so the callback will never run.
            // We have unique ownership and can therefore free it.
            drop(Box::from_raw(tx_ptr));
            vortex_err!("Failed to register CUDA stream callback: {}", err)
        })?;
    }

    Ok(rx.to_async())
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;

    use vortex::error::VortexResult;

    use super::padded_device_allocation_len;
    use crate::CudaSession;
    use crate::device_buffer::CUDF_VALIDITY_BUFFER_PADDING;
    use crate::device_buffer::cuda_backing_allocation;

    #[test]
    fn test_padded_device_allocation_len() -> VortexResult<()> {
        assert_eq!(padded_device_allocation_len::<u8>(0)?, 0);
        assert_eq!(
            padded_device_allocation_len::<u8>(1)?,
            CUDF_VALIDITY_BUFFER_PADDING
        );
        assert_eq!(
            padded_device_allocation_len::<u8>(4)?,
            CUDF_VALIDITY_BUFFER_PADDING
        );
        assert_eq!(
            padded_device_allocation_len::<u8>(5)?,
            CUDF_VALIDITY_BUFFER_PADDING
        );
        assert_eq!(
            padded_device_allocation_len::<u32>(1)?,
            CUDF_VALIDITY_BUFFER_PADDING / size_of::<u32>()
        );
        assert_eq!(
            padded_device_allocation_len::<u32>(5)?,
            CUDF_VALIDITY_BUFFER_PADDING / size_of::<u32>()
        );
        Ok(())
    }

    #[crate::test]
    async fn test_copy_to_device_preserves_visible_len_with_padding() -> VortexResult<()> {
        let ctx = CudaSession::create_execution_ctx(&crate::cuda_session())?;
        let handle = ctx.stream().copy_to_device(vec![0xab_u8])?.await?;

        assert_eq!(handle.len(), 1);
        let host = handle.try_to_host()?.await?;
        assert_eq!(host.as_slice(), &[0xab]);

        let backing = cuda_backing_allocation(&handle)?;
        assert_eq!(backing.len(), CUDF_VALIDITY_BUFFER_PADDING);
        let backing_host = backing.try_to_host()?.await?;
        assert_eq!(backing_host[0], 0xab);
        assert!(backing_host[1..].iter().all(|byte| *byte == 0));

        Ok(())
    }

    #[crate::test]
    async fn test_copy_to_device_sync_preserves_visible_len_with_padding() -> VortexResult<()> {
        let ctx = CudaSession::create_execution_ctx(&crate::cuda_session())?;
        let handle = ctx.stream().copy_to_device_sync(&[1_u8, 2, 3, 4, 5])?;

        assert_eq!(handle.len(), 5);
        let host = handle.try_to_host()?.await?;
        assert_eq!(host.as_slice(), &[1, 2, 3, 4, 5]);

        let backing = cuda_backing_allocation(&handle)?;
        assert_eq!(backing.len(), CUDF_VALIDITY_BUFFER_PADDING);
        let backing_host = backing.try_to_host()?.await?;
        assert_eq!(&backing_host[..5], &[1, 2, 3, 4, 5]);
        assert!(backing_host[5..].iter().all(|byte| *byte == 0));

        Ok(())
    }
}
