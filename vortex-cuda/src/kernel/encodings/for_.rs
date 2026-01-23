// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::PrimitiveArrayParts;
use vortex_array::buffer::BufferHandle;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_native_simd_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_fastlanes::FoRArray;
use vortex_fastlanes::FoRVTable;

use crate::CudaBufferExt;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::launch_cuda_kernel_impl;

/// CUDA decoder for frame-of-reference.
#[derive(Debug)]
pub struct FoRExecutor;

impl FoRExecutor {
    fn try_specialize(array: ArrayRef) -> Option<FoRArray> {
        array.try_into::<FoRVTable>().ok()
    }
}

#[async_trait]
impl CudaExecute for FoRExecutor {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let array = Self::try_specialize(array).ok_or_else(|| vortex_err!("Expected FoRArray"))?;

        match_each_native_simd_ptype!(array.ptype(), |P| { decode_for::<P>(array, ctx).await })
    }
}

async fn decode_for<P>(array: FoRArray, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical>
where
    P: NativePType + DeviceRepr + Send + Sync + 'static,
{
    let array_len = array.encoded().len();
    assert!(array_len > 0);

    let reference: P = array
        .reference_scalar()
        .as_primitive()
        .as_::<P>()
        .vortex_expect("Cannot have a null reference");

    // Execute child and copy to device
    let canonical = array.encoded().clone().execute_cuda(ctx).await?;
    let primitive = canonical.into_primitive();
    let PrimitiveArrayParts {
        buffer, validity, ..
    } = primitive.into_parts();

    let device_buffer: BufferHandle = if buffer.is_on_device() {
        buffer
    } else {
        ctx.move_to_device::<P>(buffer)?.await?
    };

    // Get CUDA view of the buffer
    let cuda_view = device_buffer.cuda_view::<P>()?;
    let array_len_u64 = array_len as u64;

    // Load kernel function
    let kernel_ptypes = [P::PTYPE];
    let cuda_function = ctx.load_function_ptype("for", &kernel_ptypes)?;
    let mut launch_builder = ctx.launch_builder(&cuda_function);

    // Build launch args: buffer, reference, length
    launch_builder.arg(&cuda_view);
    launch_builder.arg(&reference);
    launch_builder.arg(&array_len_u64);

    // Launch kernel
    let _cuda_events =
        launch_cuda_kernel_impl(&mut launch_builder, CU_EVENT_DISABLE_TIMING, array_len)?;

    // Build result - in-place reuses the same buffer
    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        device_buffer,
        P::PTYPE,
        validity,
    )))
}

#[cfg(test)]
#[cfg(cuda_available)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity::NonNullable;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_fastlanes::FoRArray;
    use vortex_session::VortexSession;

    use super::*;
    use crate::session::CudaSession;

    #[tokio::test]
    async fn test_cuda_for_decompression_u8() {
        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
            .vortex_expect("failed to create execution context");

        #[allow(clippy::cast_possible_truncation)]
        let input_data: Vec<u8> = (0..5000).map(|i| (i % 246) as u8).collect();

        let for_array = FoRArray::try_new(
            PrimitiveArray::new(Buffer::from(input_data), NonNullable).into_array(),
            10u8.into(),
        )
        .vortex_expect("failed to create FoR array");

        // Decode on CPU
        let cpu_result = for_array
            .to_canonical()
            .vortex_expect("CPU canonicalize failed");

        // Decode on GPU
        let gpu_result = FoRExecutor
            .execute(for_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");

        // Copy GPU result back to host for comparison
        let gpu_host = Buffer::<u8>::from_byte_buffer(
            gpu_result.into_primitive().buffer_handle().to_host().await,
        );
        let gpu_array = PrimitiveArray::new(gpu_host, NonNullable);

        assert_arrays_eq!(cpu_result.into_array(), gpu_array.into_array());
    }

    #[tokio::test]
    async fn test_cuda_for_decompression_u16() {
        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let input_data: Vec<u16> = (0..5000).map(|i| (i % 5000) as u16).collect();

        let for_array = FoRArray::try_new(
            PrimitiveArray::new(Buffer::from(input_data), NonNullable).into_array(),
            1000u16.into(),
        )
        .vortex_expect("failed to create FoR array");

        // Decode on CPU
        let cpu_result = for_array
            .to_canonical()
            .vortex_expect("CPU canonicalize failed");

        // Decode on GPU
        let gpu_result = FoRExecutor
            .execute(for_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");

        // Copy GPU result back to host for comparison
        let gpu_host = Buffer::<u16>::from_byte_buffer(
            gpu_result.into_primitive().buffer_handle().to_host().await,
        );
        let gpu_array = PrimitiveArray::new(gpu_host, NonNullable);

        assert_arrays_eq!(cpu_result.into_array(), gpu_array.into_array());
    }

    #[tokio::test]
    async fn test_cuda_for_decompression_u32() {
        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let input_data: Vec<u32> = (0..5000).map(|i| (i % 5000) as u32).collect();

        let for_array = FoRArray::try_new(
            PrimitiveArray::new(Buffer::from(input_data), NonNullable).into_array(),
            100000u32.into(),
        )
        .vortex_expect("failed to create FoR array");

        // Decode on CPU
        let cpu_result = for_array
            .to_canonical()
            .vortex_expect("CPU canonicalize failed");

        // Decode on GPU
        let gpu_result = FoRExecutor
            .execute(for_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");

        // Copy GPU result back to host for comparison
        let gpu_host = Buffer::<u32>::from_byte_buffer(
            gpu_result.into_primitive().buffer_handle().to_host().await,
        );
        let gpu_array = PrimitiveArray::new(gpu_host, NonNullable);

        assert_arrays_eq!(cpu_result.into_array(), gpu_array.into_array());
    }

    #[tokio::test]
    async fn test_cuda_for_decompression_u64() {
        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let input_data: Vec<u64> = (0..5000).map(|i| (i % 5000) as u64).collect();

        let for_array = FoRArray::try_new(
            PrimitiveArray::new(Buffer::from(input_data), NonNullable).into_array(),
            1000000u64.into(),
        )
        .vortex_expect("failed to create FoR array");

        // Decode on CPU
        let cpu_result = for_array
            .to_canonical()
            .vortex_expect("CPU canonicalize failed");

        // Decode on GPU
        let gpu_result = FoRExecutor
            .execute(for_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");

        // Copy GPU result back to host for comparison
        let gpu_host = Buffer::<u64>::from_byte_buffer(
            gpu_result.into_primitive().buffer_handle().to_host().await,
        );
        let gpu_array = PrimitiveArray::new(gpu_host, NonNullable);

        assert_arrays_eq!(cpu_result.into_array(), gpu_array.into_array());
    }
}
