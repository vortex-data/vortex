// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA stream utility functions.

use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Arc;

use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DeviceRepr;
use cudarc::driver::result::stream;
use futures::future::BoxFuture;
use kanal::Sender;
use tracing::warn;
use vortex::array::buffer::BufferHandle;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::CudaDeviceBuffer;

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
    pub(crate) fn copy_to_device<T, D>(
        &self,
        data: D,
    ) -> VortexResult<BoxFuture<'static, VortexResult<BufferHandle>>>
    where
        T: DeviceRepr + Debug + Send + Sync + 'static,
        D: AsRef<[T]> + Send + 'static,
    {
        let host_slice: &[T] = data.as_ref();
        // `device_alloc` binds the CUDA context to the current thread.
        let mut cuda_slice: CudaSlice<T> = self.device_alloc(host_slice.len())?;

        self.memcpy_htod(host_slice, &mut cuda_slice)
            .map_err(|e| vortex_err!("Failed to schedule H2D copy: {}", e))?;

        let cuda_buf = CudaDeviceBuffer::new(cuda_slice);
        let stream = Arc::clone(&self.0);

        Ok(Box::pin(async move {
            await_stream_callback(&stream).await?;

            // Keep source memory alive until copy completes.
            let _keep_alive = data;

            Ok(BufferHandle::new_device(Arc::new(cuda_buf)))
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
    pub(crate) fn copy_to_device_sync<T>(&self, data: &[T]) -> VortexResult<BufferHandle>
    where
        T: DeviceRepr + Debug + Send + Sync + 'static,
    {
        let mut cuda_slice: CudaSlice<T> = self.device_alloc(data.len())?;

        self.memcpy_htod(data, &mut cuda_slice)
            .map_err(|e| vortex_err!("Failed to schedule H2D copy: {}", e))?;

        let cuda_buf = CudaDeviceBuffer::new(cuda_slice);
        Ok(BufferHandle::new_device(Arc::new(cuda_buf)))
    }
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
