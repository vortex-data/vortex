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
use vortex_array::validity::Validity::NonNullable;
use vortex_dtype::FromPrimitiveOrF16;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_native_simd_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_fastlanes::FoRArray;

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

async fn execute_for_typed<P: DeviceRepr + NativePType + FromPrimitiveOrF16>(
    array: &FoRArray,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let reference = array
        .reference_scalar()
        .as_primitive()
        .as_::<P>()
        .vortex_expect("Cannot have a null reference");

    let encoded = array.encoded().clone();
    // Recursively decompresss the child.
    let unpacked_canonical = encoded.execute_cuda(ctx).await?;

    let unpacked_array = unpacked_canonical.as_primitive();
    let unpacked_slice = unpacked_array.as_slice::<P>();

    // TODO(0ax1): Check whether buffer is already on device.
    let device_data = ctx.to_device(unpacked_slice)?;

    let array_len = array.len() as u64;
    let _kernel_events = launch_cuda_kernel!(
        execution_ctx: ctx,
        module: "for",
        ptypes: &[array.ptype()],
        launch_args: [device_data, reference, array_len],
        // CUDA events are submitted before and after the kernel launch. This
        // enables waiting for a single kernel to finish without doing a global
        // synchronize on the stream. Timing is disabled to keep the overhead low.
        event_recording: CU_EVENT_DISABLE_TIMING,
        array_len: array.len()
    );

    let result = ctx.to_host(
        // TODO: Don't copy back after the end of each run.
        &device_data,
        // TODO: Proper alignment
        vortex_buffer::Alignment::of::<P>(),
    )?;

    let primitive = PrimitiveArray::new(result, NonNullable).reinterpret_cast(array.ptype());

    Ok(Canonical::Primitive(primitive))
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
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

        assert_eq!(
            result.as_primitive().as_slice::<u8>(),
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

        assert_eq!(
            result.as_primitive().as_slice::<u16>(),
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

        assert_eq!(
            result.as_primitive().as_slice::<u32>(),
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

        assert_eq!(
            result.as_primitive().as_slice::<u64>(),
            input_data
                .iter()
                .map(|&val| val.wrapping_add(1000000u64))
                .collect::<Vec<u64>>()
        );
    }
}
