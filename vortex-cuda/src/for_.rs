// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_native_simd_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_fastlanes::FoRArray;

use crate::CudaBufferExt;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::launch_cuda_kernel;

/// CUDA executor for frame-of-reference.
#[derive(Debug)]
pub struct ForExecutor;

#[async_trait]
impl CudaExecute for ForExecutor {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let for_array = array
            .as_any()
            .downcast_ref::<FoRArray>()
            .ok_or_else(|| vortex_err!("Array is not a FOR array"))?;

        execute_for(for_array, ctx).await
    }
}

async fn execute_for(array: &FoRArray, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical> {
    if array.is_empty() {
        return array.to_array().to_canonical();
    }

    // Excludes f16 support.
    match_each_native_simd_ptype!(array.ptype(), |T| {
        execute_for_typed::<T>(array, ctx).await
    })
}

async fn execute_for_typed<P: DeviceRepr + NativePType>(
    array: &FoRArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let reference = array
        .reference_scalar()
        .as_primitive()
        .as_::<P>()
        .vortex_expect("Cannot have a null reference");

    let encoded = array.encoded().clone().execute_cuda(ctx).await?;
    let (dtype, buffer_handle, validity, ..) = encoded.into_primitive().into_parts();

    let device_buffer_handle = if buffer_handle.is_on_device() {
        buffer_handle
    } else {
        ctx.copy_buffer_to_device_async::<P>(buffer_handle)?.await?
    };

    let cuda_view = device_buffer_handle.cuda_view::<P>()?;
    let array_len = array.len() as u64;

    // Ignore the CUDA events returned from the kernel launch, as the CUDA slice,
    // owned by the buffer handle, holds CUDA events that can be checked for completion.
    let _cuda_events = launch_cuda_kernel!(
        execution_ctx: ctx,
        module: "for",
        ptypes: &[array.ptype()],
        launch_args: [cuda_view, reference, array_len],
        // CUDA events are automatically submitted before and after the kernel launch.
        event_recording: CU_EVENT_DISABLE_TIMING,
        array_len: array.len()
    );

    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        device_buffer_handle,
        dtype.as_ptype(),
        validity,
    )))
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity::NonNullable;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_fastlanes::FoRArray;
    use vortex_session::VortexSession;

    use super::*;
    use crate::has_nvcc;
    use crate::session::CudaSession;

    #[tokio::test]
    async fn test_cuda_for_decompression_u8() {
        if !has_nvcc() {
            return;
        }

        let mut cuda_ctx = CudaSession::new_ctx(VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create u8 offset values that cycle through 0-255, creating 5000 elements
        #[allow(clippy::cast_possible_truncation)]
        let input_data: Vec<u8> = (0..5000).map(|i| (i % 256) as u8).collect();

        let for_array = FoRArray::try_new(
            PrimitiveArray::new(Buffer::from(input_data.clone()), NonNullable).into_array(),
            10u8.into(),
        )
        .vortex_expect("failed to create FoR array");

        // Decompress on the GPU.
        let result = execute_for(&for_array, &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");

        let result_buf =
            Buffer::<u8>::from_byte_buffer(result.as_primitive().buffer_handle().to_host());

        assert_eq!(
            result_buf,
            input_data
                .iter()
                .map(|&val| val.wrapping_add(10))
                .collect::<Vec<u8>>()
        );
    }

    #[tokio::test]
    async fn test_cuda_for_decompression_u16() {
        if !has_nvcc() {
            return;
        }

        let mut cuda_ctx = CudaSession::new_ctx(VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create u16 offset values that cycle through 0-5000, creating 5000 elements
        let input_data: Vec<u16> = (0..5000).map(|i| (i % 5000) as u16).collect();

        let for_array = FoRArray::try_new(
            PrimitiveArray::new(Buffer::from(input_data.clone()), NonNullable).into_array(),
            1000u16.into(),
        )
        .vortex_expect("failed to create FoR array");

        // Decompress on the GPU.
        let result = execute_for(&for_array, &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");

        let result_buf =
            Buffer::<u16>::from_byte_buffer(result.as_primitive().buffer_handle().to_host());

        assert_eq!(
            result_buf,
            input_data
                .iter()
                .map(|&val| val.wrapping_add(1000))
                .collect::<Vec<u16>>()
        );
    }

    #[tokio::test]
    async fn test_cuda_for_decompression_u32() {
        if !has_nvcc() {
            return;
        }

        let mut cuda_ctx = CudaSession::new_ctx(VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create u32 offset values that cycle through 0-5000, creating 5000 elements
        let input_data: Vec<u32> = (0..5000).map(|i| (i % 5000) as u32).collect();

        let for_array = FoRArray::try_new(
            PrimitiveArray::new(Buffer::from(input_data.clone()), NonNullable).into_array(),
            100000u32.into(),
        )
        .vortex_expect("failed to create FoR array");

        // Decompress on the GPU.
        let result = execute_for(&for_array, &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");

        let result_buf =
            Buffer::<u32>::from_byte_buffer(result.as_primitive().buffer_handle().to_host());

        assert_eq!(
            result_buf,
            input_data
                .iter()
                .map(|&val| val.wrapping_add(100000))
                .collect::<Vec<u32>>()
        );
    }

    #[tokio::test]
    async fn test_cuda_for_decompression_u64() {
        if !has_nvcc() {
            return;
        }

        let mut cuda_ctx = CudaSession::new_ctx(VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create u64 offset values that cycle through 0-5000, creating 5000 elements
        let input_data: Vec<u64> = (0..5000).map(|i| (i % 5000) as u64).collect();

        let for_array = FoRArray::try_new(
            PrimitiveArray::new(Buffer::from(input_data.clone()), NonNullable).into_array(),
            1000000u64.into(),
        )
        .vortex_expect("failed to create FoR array");

        // Decompress on the GPU.
        let result = execute_for(&for_array, &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");

        let result_buf =
            Buffer::<u64>::from_byte_buffer(result.as_primitive().buffer_handle().to_host());

        assert_eq!(
            result_buf,
            input_data
                .iter()
                .map(|&val| val.wrapping_add(1000000u64))
                .collect::<Vec<u64>>()
        );
    }
}
