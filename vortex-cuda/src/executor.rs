// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::CudaFunction;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchArgs;
use cudarc::driver::ValidAsZeroBits;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::VortexSessionExecute;
use vortex_buffer::Alignment;
use vortex_buffer::Buffer;
use vortex_buffer::BufferMut;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_session::VortexSession;

use crate::CudaSession;
use crate::session::CudaSessionExt;

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
#[macro_export]
macro_rules! launch_cuda_kernel {
    (
        execution_ctx: $ctx:expr,
        module: $module:expr,
        ptypes: $ptypes:expr,
        launch_args: [$($arg:expr),* $(,)?],
        array_len: $len:expr
    ) => {{
        let cuda_function = $ctx.load_function($module, $ptypes)?;
        let mut launch_builder = $ctx.launch_builder(&cuda_function);

        // Unroll launch builder arguments.
        $(
            launch_builder.arg(&$arg);
        )*

        let num_chunks = u32::try_from($len.div_ceil(2048))
            .vortex_expect("Too many elements for grid");

        let config = cudarc::driver::LaunchConfig {
            grid_dim: (num_chunks, 1, 1),
            block_dim: (64, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            launch_builder
                .launch(config)
                .map_err(|e| vortex_err!("Failed to launch kernel: {}", e))?
        };
    }};
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
    pub fn alloc<T: DeviceRepr + ValidAsZeroBits>(&self, len: usize) -> VortexResult<CudaSlice<T>> {
        // SAFETY: No safety guarantees for allocations on the GPU.
        unsafe {
            self.stream
                // Note that alloc is async in case the device and driver support this.
                //
                // The condition for alloc to be async is support for memory pools:
                // `CU_DEVICE_ATTRIBUTE_MEMORY_POOLS_SUPPORTED`. Any kernel
                // submitted to the stream after alloc can safely use the memory,
                // as operations on the stream are ordered sequentially.
                .alloc::<T>(len)
                .map_err(|e| vortex_err!("Failed to allocate device memory: {}", e))
        }
    }

    /// Copies data from host to device.
    pub fn to_device<T: DeviceRepr>(&self, data: &[T]) -> VortexResult<CudaSlice<T>> {
        // TODO(0ax1): Make the memcopy to device async. Even though `memcpy_htod`
        // uses into `memcpy_htod_async`, it implicitly calls synchronize on the
        // stream when dropping the `SyncOnDrop` `_record_dst` event at the end
        // of the function.
        self.stream
            .clone_htod(data)
            .map_err(|e| vortex_err!("Failed to copy to device: {}", e))
    }

    /// Copies data from device to host.
    ///
    /// Returns a `Buffer<T>` with the specified alignment.
    pub fn to_host<T: DeviceRepr>(
        &self,
        buffer: &CudaSlice<T>,
        alignment: Alignment,
    ) -> VortexResult<Buffer<T>> {
        let len = buffer.len();
        let mut host_buffer = BufferMut::<T>::with_capacity_aligned(len, alignment);

        // TODO(0ax1): Make the memcopy to host async. Even though `memcpy_dtoh`
        // uses into `memcpy_dtoh_async`, it implicitly calls synchronize on the
        // stream when dropping the `SyncOnDrop` `_record_dst` event at the end
        // of the function.
        self.stream
            .memcpy_dtoh(buffer, unsafe {
                // SAFETY: We allocated with sufficient capacity and fill the entire buffer.
                host_buffer.set_len(len);
                host_buffer.as_mut_slice()
            })
            .map_err(|e| vortex_err!("Failed to copy from device: {}", e))?;

        Ok(host_buffer.freeze())
    }

    /// Synchronizes the stream
    ///
    /// On `synchronize` the host waits for all pending operations of the stream to complete.
    #[cfg(test)]
    pub fn synchronize(&self) -> VortexResult<()> {
        self.stream
            .synchronize()
            .map_err(|e| vortex_err!("Failed to synchronize device: {}", e))
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
            return Ok(self.to_canonical());
        }

        let Some(support) = ctx.cuda_session.kernel(&self.encoding_id()) else {
            tracing::debug!(
                encoding = %self.encoding().id(),
                "No CUDA support registered for encoding, falling back to CPU execution"
            );
            let mut array_ctx = ctx.vortex_session.create_execution_ctx();
            return self.execute(&mut array_ctx);
        };

        tracing::debug!(
            encoding = %self.encoding().id(),
            "Executing array on CUDA device"
        );

        support.execute(self, ctx).await
    }
}
