// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::mem::size_of;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaFunction;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DevicePtrMut;
use cudarc::driver::DeviceRepr;
use cudarc::driver::DriverError;
use cudarc::driver::LaunchArgs;
use cudarc::driver::result;
use cudarc::driver::result::memcpy_htod_async;
use cudarc::driver::sys;
use cudarc::driver::sys::CUevent_flags;
use futures::future::BoxFuture;
use kanal::Sender;
use result::stream;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::VortexSessionExecute;
use vortex_array::buffer::BufferHandle;
use vortex_buffer::Buffer;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::CudaDeviceBuffer;
use crate::CudaSession;
use crate::session::CudaSessionExt;

/// Registers a callback and asynchronously waits for its completion.
///
/// This function can be used to asynchronously wait for events previously
/// submitted to the stream to complete, e.g. async device buffer allocations.
///
/// Note: This is not equivalent to calling sync on a stream but only awaits
/// the registered callback to complete.
///
/// # Arguments
///
/// * `stream` - The CUDA stream to wait on
pub async fn await_stream_callback(stream: &CudaStream) -> Result<(), DriverError> {
    let rx = register_stream_callback(stream)?;

    rx.recv()
        .await
        .map_err(|_| DriverError(sys::CUresult::CUDA_ERROR_UNKNOWN))
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
fn register_stream_callback(stream: &CudaStream) -> Result<kanal::AsyncReceiver<()>, DriverError> {
    let (tx, rx) = kanal::bounded::<()>(1);

    // There are 2 different scenarios how `tx` gets freed. When the callback
    // is invoked or during cleanup in case the registration fails.
    let tx_ptr = Box::into_raw(Box::new(tx));

    /// Called from CUDA driver thread when all preceding work on the stream completes.
    unsafe extern "C" fn callback(user_data: *mut std::ffi::c_void) {
        // SAFETY: The memory of `tx` is manually managed has not been freed
        // before. We have unique ownership and can therefore free it.
        let tx = unsafe { Box::from_raw(user_data as *mut Sender<()>) };

        // Blocking send as we're in a callback invoked by the CUDA driver.
        #[expect(clippy::expect_used)]
        tx.send(())
            // A send should never fail. Panic otherwise.
            .expect("CUDA callback receiver dropped unexpectedly");
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
        .inspect_err(|_| {
            // SAFETY: Registration failed, so callback will never run.
            // Therefore, we need to free the `user_data` passed to the
            // callback in the error case.
            drop(Box::from_raw(tx_ptr));
        })?;
    }

    Ok(rx.to_async())
}

/// CUDA kernel events recorded before and after kernel launch.
#[derive(Debug)]
pub struct CudaKernelEvents {
    /// Event recorded before kernel launch.
    pub before_launch: CudaEvent,
    /// Event recorded after kernel launch.
    pub after_launch: CudaEvent,
}

/// Convenience macro to launch a CUDA kernel.
///
/// The kernel gets launched on the stream of the execution context.
///
/// The kernel launch config:
/// LaunchConfig {
///     grid_dim: (array.len() / 2048, 1, 1),
///     block_dim: (64, 1, 1),
///     shared_mem_bytes: 0,
/// };
/// 64 threads are used per block which corresponds to 2 warps.
/// Each block handles 2048 elements. Each thread handles 32 elements.
/// The last block and thread are allowed to have less elements.
///
/// Note: A macro is necessary to unroll the launch builder arguments.
///
/// # Returns
///
/// A pair of CUDA events submitted before and after the kernel.
/// Depending on `CUevent_flags` these events can contain timestamps. Use
/// `CU_EVENT_DISABLE_TIMING` for minimal overhead and `CU_EVENT_DEFAULT` to
/// enable timestamps.
#[macro_export]
macro_rules! launch_cuda_kernel {
    (
        execution_ctx: $ctx:expr,
        module: $module:expr,
        ptypes: $ptypes:expr,
        launch_args: [$($arg:expr),* $(,)?],
        event_recording: $event_recording:expr,
        array_len: $len:expr
    ) => {{
        let cuda_function = $ctx.load_function($module, $ptypes)?;
        let mut launch_builder = $ctx.launch_builder(&cuda_function);

        $(
            launch_builder.arg(&$arg);
        )*

        $crate::executor::launch_cuda_kernel_impl(&mut launch_builder, $event_recording, $len)?
    }};
}

/// Launches a CUDA kernel with the passed launch builder.
///
/// # Arguments
///
/// * `launch_builder` - Configured launch builder
/// * `array_len` - Length of the array to process
///
/// # Returns
///
/// A pair of CUDA events submitted before and after the kernel.
/// Depending on `CUevent_flags` these events can contain timestamps. Use
/// `CU_EVENT_DISABLE_TIMING` for minimal overhead and `CU_EVENT_DEFAULT` to
/// enable timestamps.
pub fn launch_cuda_kernel_impl(
    launch_builder: &mut LaunchArgs,
    event_flags: CUevent_flags,
    array_len: usize,
) -> VortexResult<CudaKernelEvents> {
    let num_chunks = u32::try_from(array_len.div_ceil(2048))?;

    let config = cudarc::driver::LaunchConfig {
        grid_dim: (num_chunks, 1, 1),
        block_dim: (64, 1, 1),
        shared_mem_bytes: 0,
    };

    launch_builder.record_kernel_launch(event_flags);

    unsafe {
        launch_builder
            .launch(config)
            .map_err(|e| vortex_err!("Failed to launch kernel: {}", e))
            .and_then(|events| {
                events
                    .ok_or_else(|| vortex_err!("CUDA events not recorded"))
                    .map(|(before_launch, after_launch)| CudaKernelEvents {
                        before_launch,
                        after_launch,
                    })
            })
    }
}

/// CUDA execution context.
///
/// Provides access to the CUDA context and stream for kernel execution.
/// Handles memory allocation and data transfers between host and device.
pub struct CudaExecutionCtx {
    stream: Arc<CudaStream>,
    vortex_session: VortexSession,
    cuda_session: CudaSession,
}

impl CudaExecutionCtx {
    /// Creates a new CUDA execution context.
    pub(crate) fn new(stream: Arc<CudaStream>, vortex_session: VortexSession) -> Self {
        let cuda_session = vortex_session.cuda_session().clone();
        Self {
            stream,
            vortex_session,
            cuda_session,
        }
    }

    /// Allocates a typed buffer on the GPU.
    ///
    /// Note: Allocation is async in case the CUDA driver supports this.
    ///
    /// The condition for alloc to be async is support for memory pools:
    /// `CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED`.
    ///
    /// Any kernel submitted to the stream after alloc can safely use the
    /// memory, as operations on the stream are ordered sequentially.
    pub fn device_alloc<T: DeviceRepr>(&self, len: usize) -> VortexResult<CudaSlice<T>> {
        // SAFETY: No safety guarantees for allocations on the GPU.
        unsafe {
            self.stream
                .alloc::<T>(len)
                .map_err(|e| vortex_err!("Failed to allocate device memory: {}", e))
        }
    }

    /// Loads a CUDA kernel function by module name and ptype(s).
    ///
    /// # Arguments
    ///
    /// * `module_name` - Name of the module (`kernels/{module_name}.ptx`)
    /// * `ptypes` - List of ptype strings for the kernel name
    ///
    /// # Errors
    ///
    /// Returns an error if kernel loading fails.
    pub fn load_function(&self, module_name: &str, ptypes: &[PType]) -> VortexResult<CudaFunction> {
        self.cuda_session.load_function(module_name, ptypes)
    }

    /// Returns a launch builder for a CUDA kernel function.
    ///
    /// Arguments can be added to the kernel launch with `.arg(buffer)`.
    ///
    /// # Arguments
    ///
    /// * `func` - CUDA kernel function to launch
    pub fn launch_builder<'a>(&'a self, func: &'a CudaFunction) -> LaunchArgs<'a> {
        self.stream.launch_builder(func)
    }

    /// Copies host data to the device, returning a [`CudaDeviceBuffer`].
    pub fn copy_buffer_to_device<T: DeviceRepr>(
        &self,
        data: &[T],
    ) -> VortexResult<CudaDeviceBuffer<T>> {
        let cuda_slice = self
            .stream
            .clone_htod(data)
            .map_err(|e| vortex_err!("Failed to copy to device: {}", e))?;
        Ok(CudaDeviceBuffer::new(cuda_slice))
    }

    /// Copies a pinned host buffer to the device asynchronously.
    ///
    /// Allocates device memory, schedules an async copy, and returns a future
    /// that completes when the copy is finished.
    ///
    /// # Arguments
    ///
    /// * `handle` - The host buffer to copy. Must be a host buffer (not already on device).
    ///
    /// # Safety
    ///
    /// The returned future captures the source `BufferHandle` to keep the host
    /// memory alive until the copy completes.
    pub fn copy_buffer_to_device_async<T: DeviceRepr + Send + Sync + 'static>(
        &self,
        handle: BufferHandle,
    ) -> VortexResult<BoxFuture<'static, VortexResult<BufferHandle>>> {
        let host_buffer = handle
            .as_host_opt()
            .ok_or_else(|| vortex_err!("Buffer is neither on host nor device"))?;

        let mut cuda_slice: CudaSlice<T> = self.device_alloc(host_buffer.len() / size_of::<T>())?;
        let device_ptr = cuda_slice.device_ptr_mut(&self.stream).0;

        let typed_buffer: Buffer<T> = Buffer::from_byte_buffer(host_buffer.clone());
        let src_slice: &[T] = typed_buffer.as_slice();

        unsafe {
            memcpy_htod_async(device_ptr, src_slice, self.stream.cu_stream())
                .map_err(|e| vortex_err!("Failed to schedule async copy to device: {}", e))?;
        }

        let cuda_buf = CudaDeviceBuffer::new(cuda_slice);
        let stream = Arc::clone(&self.stream);

        Ok(Box::pin(async move {
            // Await async copy completion using callback-based async wait.
            await_stream_callback(&stream)
                .await
                .map_err(|e| vortex_err!("CUDA stream wait failed: {}", e))?;

            // Keep source memory alive until copy completes.
            let _keep_alive = handle;

            Ok(BufferHandle::new_device(Arc::new(cuda_buf)))
        }))
    }
}

/// Support trait for CUDA-accelerated decompression of arrays.
#[async_trait]
pub trait CudaExecute: 'static + Send + Sync + Debug {
    /// Executes the array on CUDA, returning a canonical array.
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails on the GPU.
    async fn execute(&self, array: ArrayRef, ctx: &mut CudaExecutionCtx)
    -> VortexResult<Canonical>;
}

/// Extension trait for executing arrays on CUDA.
#[async_trait]
pub trait CudaArrayExt: Array {
    /// Recursively executes the array on CUDA, returning a canonical array.
    ///
    /// If no CUDA support is registered for the encoding, falls back to CPU execution
    /// and logs a debug message.
    async fn execute_cuda(self, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical>;
}

#[async_trait]
impl CudaArrayExt for ArrayRef {
    async fn execute_cuda(self, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical> {
        if self.is_canonical() {
            return self.to_canonical();
        }

        let Some(support) = ctx.cuda_session.kernel(&self.encoding_id()) else {
            tracing::debug!(
                encoding = %self.encoding_id(),
                "No CUDA support registered for encoding, falling back to CPU execution"
            );
            let mut array_ctx = ctx.vortex_session.create_execution_ctx();
            return self.execute(&mut array_ctx);
        };

        tracing::debug!(
            encoding = %self.encoding_id(),
            "Executing array on CUDA device"
        );

        support.execute(self, ctx).await
    }
}
