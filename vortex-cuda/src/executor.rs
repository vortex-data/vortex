// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;
use std::time::Duration;

use async_trait::async_trait;
use cudarc::driver::CudaEvent;
use cudarc::driver::CudaFunction;
use cudarc::driver::CudaSlice;
use cudarc::driver::CudaStream;
use cudarc::driver::DeviceRepr;
use cudarc::driver::LaunchArgs;
use futures::future::BoxFuture;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::StructArray;
use vortex_array::arrays::StructArrayParts;
use vortex_array::arrays::StructVTable;
use vortex_array::buffer::BufferHandle;
use vortex_dtype::PType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::CudaSession;
use crate::ExportDeviceArray;
use crate::session::CudaSessionExt;
use crate::stream::VortexCudaStream;

/// CUDA kernel events recorded before and after kernel launch.
#[derive(Debug)]
pub struct CudaKernelEvents {
    /// Event recorded before kernel launch.
    pub before_launch: CudaEvent,
    /// Event recorded after kernel launch.
    pub after_launch: CudaEvent,
}

impl CudaKernelEvents {
    pub fn duration(&self) -> VortexResult<Duration> {
        self.before_launch
            .elapsed_ms(&self.after_launch) // synchronizes
            .map_err(|e| vortex_err!("failed to get elapsed time: {}", e))
            .map(|f| Duration::from_secs_f32(f / 1000.0))
    }
}

/// CUDA execution context.
///
/// Provides access to the CUDA context and stream for kernel execution.
/// Handles memory allocation and data transfers between host and device.
pub struct CudaExecutionCtx {
    stream: VortexCudaStream,
    ctx: ExecutionCtx,
    cuda_session: CudaSession,
}

impl CudaExecutionCtx {
    /// Creates a new CUDA execution context.
    pub(crate) fn new(stream: VortexCudaStream, ctx: ExecutionCtx) -> Self {
        let cuda_session = ctx.session().cuda_session().clone();
        Self {
            stream,
            ctx,
            cuda_session,
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
    pub fn load_function_ptype(
        &self,
        module_name: &str,
        ptypes: &[PType],
    ) -> VortexResult<CudaFunction> {
        let type_suffixes: Vec<String> = ptypes.iter().map(|ptype| ptype.to_string()).collect();
        self.load_function(
            module_name,
            type_suffixes
                .iter()
                .map(|t| t.as_str())
                .collect::<Vec<_>>()
                .as_slice(),
        )
    }

    /// Loads a CUDA kernel function by module name and type suffixes.
    ///
    /// This is a lower-level version of `load_function` that accepts string suffixes
    /// directly, useful for types that don't have a `PType` (e.g., i128, i256).
    ///
    /// # Arguments
    ///
    /// * `module_name` - Name of the module (`kernels/{module_name}.ptx`)
    /// * `type_suffixes` - List of type suffix strings for the kernel name
    ///
    /// # Errors
    ///
    /// Returns an error if kernel loading fails.
    pub fn load_function(
        &self,
        module_name: &str,
        type_suffixes: &[&str],
    ) -> VortexResult<CudaFunction> {
        self.cuda_session
            .load_function_with_suffixes(module_name, type_suffixes)
    }

    /// Returns a launch builder for a CUDA kernel function.
    ///
    /// Arguments can be added to the kernel launch with `.arg(buffer)`.
    ///
    /// # Arguments
    ///
    /// * `func` - CUDA kernel function to launch
    pub fn launch_builder<'a>(&'a self, func: &'a CudaFunction) -> LaunchArgs<'a> {
        self.stream.0.launch_builder(func)
    }

    /// See `VortexCudaStream::device_alloc`.
    pub fn device_alloc<T: DeviceRepr + Send + Sync + 'static>(
        &self,
        len: usize,
    ) -> VortexResult<CudaSlice<T>> {
        self.stream.device_alloc(len)
    }

    /// See `VortexCudaStream::copy_to_device`.
    pub fn copy_to_device<T, D>(
        &self,
        data: D,
    ) -> VortexResult<BoxFuture<'static, VortexResult<BufferHandle>>>
    where
        T: DeviceRepr + Debug + Send + Sync + 'static,
        D: AsRef<[T]> + Send + 'static,
    {
        self.stream.copy_to_device(data)
    }

    /// See `VortexCudaStream::move_to_device`.
    pub fn move_to_device(
        &self,
        handle: BufferHandle,
    ) -> VortexResult<BoxFuture<'static, VortexResult<BufferHandle>>> {
        self.stream.move_to_device(handle)
    }

    /// Returns a reference to the underlying CUDA stream.
    pub fn stream(&self) -> &Arc<CudaStream> {
        &self.stream.0
    }

    /// Get a handle to the exporter that can convert arrays into `ArrowDeviceArray`.
    pub fn exporter(&self) -> &Arc<dyn ExportDeviceArray> {
        self.cuda_session.export_device_array()
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
    #[allow(clippy::unwrap_in_result, clippy::unwrap_used)]
    async fn execute_cuda(self, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical> {
        if self.encoding_id() == StructVTable::ID {
            let len = self.len();
            let StructArrayParts {
                fields,
                struct_fields,
                validity,
                ..
            } = self.try_into::<StructVTable>().unwrap().into_parts();

            let mut cuda_fields = Vec::with_capacity(fields.len());
            for field in fields.iter() {
                cuda_fields.push(field.clone().execute_cuda(ctx).await?.into_array());
            }

            return Ok(Canonical::Struct(StructArray::new(
                struct_fields.names().clone(),
                cuda_fields,
                len,
                validity,
            )));
        }

        if self.is_canonical() || self.is_empty() {
            return self.execute(&mut ctx.ctx);
        }

        let Some(support) = ctx.cuda_session.kernel(&self.encoding_id()) else {
            tracing::debug!(
                encoding = %self.encoding_id(),
                "No CUDA support registered for encoding, falling back to CPU execution"
            );
            return self.execute(&mut ctx.ctx);
        };

        tracing::debug!(
            encoding = %self.encoding_id(),
            "Executing array on CUDA device"
        );

        support.execute(self, ctx).await
    }
}

#[cfg(feature = "_test-harness")]
impl CudaExecutionCtx {
    pub fn synchronize_stream(&self) -> VortexResult<()> {
        self.stream
            .0
            .synchronize()
            .map_err(|e| vortex_err!("cuda error: {e}"))
    }
}
