// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! CUDA executor for the posterize scalar function.

use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::LaunchConfig;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex::array::Array;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::PrimitiveArrayParts;
use vortex::array::arrays::ScalarFnVTable as ScalarFnArrayVTable;
use vortex::array::buffer::BufferHandle;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::scalar_fn::posterize::Posterize;

/// CUDA executor for the posterize scalar function.
///
/// Quantizes uint8 values to a fixed number of evenly spaced levels on the GPU.
#[derive(Debug)]
pub(crate) struct PosterizeCudaExecutor;

#[async_trait]
impl CudaExecute for PosterizeCudaExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        // Downcast to ScalarFnArray and extract posterize options
        let scalar_fn_array = array
            .as_opt::<ScalarFnArrayVTable>()
            .ok_or_else(|| vortex_err!("Expected ScalarFnArray for posterize"))?;

        let levels = scalar_fn_array
            .scalar_fn()
            .as_opt::<Posterize>()
            .ok_or_else(|| vortex_err!("Expected Posterize scalar function"))?
            .levels;

        let child = scalar_fn_array.children()[0].clone();
        let array_len = child.len();

        // Recursively execute the child on GPU
        let canonical = child.execute_cuda(ctx).await?;
        let primitive = canonical.into_primitive();
        let PrimitiveArrayParts {
            buffer, validity, ..
        } = primitive.into_parts();

        let input_buffer = ctx.ensure_on_device(buffer).await?;
        let input_ptr = input_buffer.cuda_device_ptr()?;

        // Allocate output buffer on device
        let output_slice: cudarc::driver::CudaSlice<u8> = ctx.device_alloc(array_len)?;
        let output_handle = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(output_slice)));
        let output_ptr = output_handle.cuda_device_ptr()?;

        // Load and launch posterize kernel
        let func = ctx.load_function("posterize", &[])?;
        let len_u32 = u32::try_from(array_len)?;

        let threads = 256u32;
        let blocks = len_u32.div_ceil(threads);
        let config = LaunchConfig {
            grid_dim: (blocks, 1, 1),
            block_dim: (threads, 1, 1),
            shared_mem_bytes: 0,
        };

        ctx.launch_kernel_config(&func, config, array_len, |args| {
            args.arg(&input_ptr)
                .arg(&output_ptr)
                .arg(&len_u32)
                .arg(&levels);
        })?;

        Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
            output_handle,
            vortex::dtype::PType::U8,
            validity,
        )))
    }
}
