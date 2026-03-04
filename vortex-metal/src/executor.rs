// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::ptr::NonNull;

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::MTLBuffer;
use objc2_metal::MTLCommandBuffer;
use objc2_metal::MTLCommandEncoder;
use objc2_metal::MTLCommandQueue;
use objc2_metal::MTLComputeCommandEncoder;
use objc2_metal::MTLComputePipelineState;
use objc2_metal::MTLDevice;
use objc2_metal::MTLResourceOptions;
use objc2_metal::MTLSize;
use tracing::debug;
use tracing::trace;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::DynArray;
use vortex::array::ExecutionCtx;
use vortex::array::IntoArray;
use vortex::array::arrays::StructArray;
use vortex::array::arrays::StructArrayParts;
use vortex::array::arrays::StructVTable;
use vortex::array::buffer::BufferHandle;
use vortex::buffer::Alignment;
use vortex::buffer::ByteBuffer;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::MetalDeviceBuffer;
use crate::MetalSession;

/// Metal execution context.
///
/// Provides access to the Metal device and command buffer for kernel execution.
/// Handles memory allocation and data transfers between host and device.
pub struct MetalExecutionCtx {
    /// The Metal session
    metal_session: MetalSession,
    /// CPU execution context for fallback
    ctx: ExecutionCtx,
    /// Current command buffer
    command_buffer: Option<Retained<ProtocolObject<dyn MTLCommandBuffer>>>,
}

impl MetalExecutionCtx {
    /// Creates a new Metal execution context.
    pub(crate) fn new(metal_session: MetalSession, ctx: ExecutionCtx) -> VortexResult<Self> {
        Ok(Self {
            metal_session,
            ctx,
            command_buffer: None,
        })
    }

    /// Get a mutable handle to the CPU execution context.
    pub fn execution_ctx(&mut self) -> &mut ExecutionCtx {
        &mut self.ctx
    }

    /// Returns a reference to the Metal session.
    pub fn session(&self) -> &MetalSession {
        &self.metal_session
    }

    /// Returns or creates a command buffer for the current execution.
    pub fn command_buffer(&mut self) -> VortexResult<&ProtocolObject<dyn MTLCommandBuffer>> {
        if self.command_buffer.is_none() {
            let cmd_buffer = self
                .metal_session
                .command_queue()
                .commandBuffer()
                .ok_or_else(|| vortex_err!("Failed to create Metal command buffer"))?;
            self.command_buffer = Some(cmd_buffer);
        }
        Ok(self
            .command_buffer
            .as_ref()
            .vortex_expect("command buffer should exist"))
    }

    /// Commits the current command buffer and waits for completion.
    pub fn commit_and_wait(&mut self) -> VortexResult<()> {
        if let Some(cmd_buffer) = self.command_buffer.take() {
            cmd_buffer.commit();
            cmd_buffer.waitUntilCompleted();
        }
        Ok(())
    }

    /// Loads a compute pipeline state for a kernel function.
    ///
    /// # Arguments
    ///
    /// * `module_name` - Name of the shader module
    /// * `function_name` - Name of the kernel function
    pub fn load_pipeline(
        &self,
        module_name: &str,
        function_name: &str,
    ) -> VortexResult<Retained<ProtocolObject<dyn MTLComputePipelineState>>> {
        self.metal_session
            .library_loader()
            .load_pipeline(module_name, function_name)
    }

    /// Allocates a buffer on the GPU.
    ///
    /// # Arguments
    ///
    /// * `len` - Size in bytes
    /// * `alignment` - Required alignment
    #[allow(dead_code)]
    pub fn device_alloc(
        &self,
        len: usize,
        alignment: Alignment,
    ) -> VortexResult<MetalDeviceBuffer> {
        // Use shared storage mode for Apple Silicon unified memory
        let options = MTLResourceOptions::StorageModeShared;

        let buffer = self
            .metal_session
            .device()
            .newBufferWithLength_options(len, options)
            .ok_or_else(|| vortex_err!("Failed to allocate Metal buffer of {} bytes", len))?;

        Ok(MetalDeviceBuffer::new(buffer, alignment))
    }

    /// Copies host data to the device.
    ///
    /// On Apple Silicon with unified memory, this creates a buffer with the data
    /// directly accessible to both CPU and GPU.
    pub fn copy_to_device(&self, data: &ByteBuffer) -> VortexResult<MetalDeviceBuffer> {
        // Use shared storage mode for Apple Silicon unified memory
        let options = MTLResourceOptions::StorageModeShared;

        // SAFETY: We're passing a valid pointer to data that will be copied.
        // The Metal API signature requires NonNull but doesn't mutate the source.
        #[allow(clippy::as_ptr_cast_mut)]
        let ptr = NonNull::new(data.as_ptr() as *mut std::ffi::c_void)
            .ok_or_else(|| vortex_err!("Null pointer passed to copy_to_device"))?;

        // SAFETY: newBufferWithBytes_length_options copies data from the pointer,
        // and we've verified the pointer is valid and the data is accessible.
        let buffer = unsafe {
            self.metal_session
                .device()
                .newBufferWithBytes_length_options(ptr, data.len(), options)
        }
        .ok_or_else(|| vortex_err!("Failed to create Metal buffer with data"))?;

        Ok(MetalDeviceBuffer::new(buffer, data.alignment()))
    }

    /// Ensures a buffer is resident on the device, copying from host if necessary.
    ///
    /// If the buffer is already on the device it is returned as-is. Otherwise
    /// copies from host to device.
    pub fn ensure_on_device(&self, handle: BufferHandle) -> VortexResult<BufferHandle> {
        if handle.is_on_device() {
            return Ok(handle);
        }

        let host_buffer = handle
            .as_host_opt()
            .ok_or_else(|| vortex_err!("Buffer is not on host"))?;

        let device_buffer = self.copy_to_device(host_buffer)?;
        Ok(device_buffer.into_buffer_handle())
    }

    /// Dispatches a compute kernel with the given pipeline state.
    ///
    /// # Arguments
    ///
    /// * `pipeline` - The compute pipeline state
    /// * `buffers` - List of (buffer, offset) pairs to bind
    /// * `constants` - Raw bytes to set as constant data at index
    /// * `array_len` - Number of elements to process
    pub fn dispatch_kernel(
        &mut self,
        pipeline: &ProtocolObject<dyn MTLComputePipelineState>,
        buffers: &[(&ProtocolObject<dyn MTLBuffer>, usize)],
        constants: &[(&[u8], usize)],
        array_len: usize,
    ) -> VortexResult<()> {
        let cmd_buffer = self.command_buffer()?;

        let encoder = cmd_buffer
            .computeCommandEncoder()
            .ok_or_else(|| vortex_err!("Failed to create compute command encoder"))?;

        encoder.setComputePipelineState(pipeline);

        // Bind buffers
        // SAFETY: We're passing valid Metal buffer references with valid offsets
        for (idx, (buffer, offset)) in buffers.iter().enumerate() {
            unsafe {
                encoder.setBuffer_offset_atIndex(Some(*buffer), *offset, idx);
            }
        }

        // Set constant data
        // SAFETY: We're passing valid data pointers for constant buffer data.
        // The Metal API signature requires NonNull but doesn't mutate the source.
        #[allow(clippy::as_ptr_cast_mut)]
        for (data, index) in constants {
            if let Some(ptr) = NonNull::new(data.as_ptr() as *mut std::ffi::c_void) {
                unsafe {
                    encoder.setBytes_length_atIndex(ptr, data.len(), *index);
                }
            }
        }

        // Calculate grid and threadgroup sizes
        let thread_execution_width = pipeline.threadExecutionWidth();
        let max_threads_per_threadgroup = pipeline.maxTotalThreadsPerThreadgroup();

        // Use a 1D grid
        let threads_per_threadgroup = MTLSize {
            width: max_threads_per_threadgroup.min(thread_execution_width * 4),
            height: 1,
            depth: 1,
        };

        let grid_size = MTLSize {
            width: array_len,
            height: 1,
            depth: 1,
        };

        encoder.dispatchThreads_threadsPerThreadgroup(grid_size, threads_per_threadgroup);
        encoder.endEncoding();

        Ok(())
    }
}

/// Support trait for Metal-accelerated decompression of arrays.
///
/// Unlike the CUDA executor, Metal execution is synchronous since Apple Silicon
/// uses unified memory and we wait for GPU completion after each kernel.
pub trait MetalExecute: 'static + Send + Sync + Debug {
    /// Executes the array on Metal, returning a canonical array.
    ///
    /// # Errors
    ///
    /// Returns an error if execution fails on the GPU.
    fn execute(&self, array: ArrayRef, ctx: &mut MetalExecutionCtx) -> VortexResult<Canonical>;
}

/// Extension trait for executing arrays on Metal.
pub trait MetalArrayExt: DynArray {
    /// Recursively walks the encoding tree, dispatching each layer to its
    /// registered [`MetalExecute`] implementation and returning a canonical array
    /// on the device.
    ///
    /// Falls back to CPU execution if no Metal support is registered for the
    /// encoding.
    fn execute_metal(self, ctx: &mut MetalExecutionCtx) -> VortexResult<Canonical>;
}

impl MetalArrayExt for ArrayRef {
    #[allow(clippy::unwrap_in_result, clippy::unwrap_used)]
    fn execute_metal(self, ctx: &mut MetalExecutionCtx) -> VortexResult<Canonical> {
        // Handle struct arrays specially - recurse into fields
        if self.encoding_id() == StructVTable::ID {
            let len = self.len();
            let StructArrayParts {
                fields,
                struct_fields,
                validity,
                ..
            } = self.try_into::<StructVTable>().unwrap().into_parts();

            let mut metal_fields = Vec::with_capacity(fields.len());
            for field in fields.iter() {
                metal_fields.push(field.clone().execute_metal(ctx)?.into_array());
            }

            return Ok(Canonical::Struct(StructArray::new(
                struct_fields.names().clone(),
                metal_fields,
                len,
                validity,
            )));
        }

        // Skip execution for canonical or empty arrays
        if self.is_canonical() || self.is_empty() {
            trace!(encoding = ?self.encoding_id(), "skipping canonical");
            return self.execute(&mut ctx.ctx);
        }

        // Look up Metal kernel for this encoding
        let Some(support) = ctx.metal_session.kernel(&self.encoding_id()) else {
            debug!(
                encoding = %self.encoding_id(),
                "No Metal support registered for encoding, falling back to CPU execution"
            );
            return self.execute(&mut ctx.ctx);
        };

        debug!(
            encoding = %self.encoding_id(),
            "Executing array on Metal device"
        );

        support.execute(self, ctx)
    }
}

/// Extension trait for copying canonical arrays from device to host.
pub trait CanonicalMetalExt {
    /// Copies all device buffers in the canonical array to host memory.
    fn into_host(self) -> VortexResult<Canonical>;
}

impl CanonicalMetalExt for Canonical {
    fn into_host(self) -> VortexResult<Canonical> {
        // For now, just convert to canonical which will copy buffers
        match self {
            Canonical::Primitive(arr) => {
                let parts = arr.into_parts();
                let host_buffer = parts.buffer.try_into_host_sync()?;
                Ok(Canonical::Primitive(
                    vortex::array::arrays::PrimitiveArray::from_buffer_handle(
                        BufferHandle::new_host(host_buffer),
                        parts.ptype,
                        parts.validity,
                    ),
                ))
            }
            other => Ok(other),
        }
    }
}
