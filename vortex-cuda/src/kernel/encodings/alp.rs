// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use cudarc::driver::sys::CUevent_flags::CU_EVENT_DISABLE_TIMING;
use vortex_alp::ALPArray;
use vortex_alp::ALPFloat;
use vortex_alp::ALPVTable;
use vortex_alp::match_each_alp_float_ptype;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::PrimitiveArrayParts;
use vortex_array::buffer::BufferHandle;
use vortex_dtype::NativePType;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::launch_cuda_kernel_impl;

/// CUDA decoder for ALP (Adaptive Lossless floating-Point) decompression.
#[derive(Debug)]
pub struct ALPExecutor;

impl ALPExecutor {
    fn try_specialize(array: ArrayRef) -> Option<ALPArray> {
        array.try_into::<ALPVTable>().ok()
    }
}

#[async_trait]
impl CudaExecute for ALPExecutor {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let array = Self::try_specialize(array).ok_or_else(|| vortex_err!("Expected ALPArray"))?;

        match_each_alp_float_ptype!(array.ptype(), |A| { decode_alp::<A>(array, ctx).await })
    }
}

async fn decode_alp<A>(array: ALPArray, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical>
where
    A: ALPFloat + NativePType + DeviceRepr + Send + Sync + 'static,
    A::ALPInt: NativePType + DeviceRepr + Send + Sync + 'static,
{
    let array_len = array.encoded().len();
    assert!(array_len > 0);

    // Get the exponent factors from the lookup tables.
    let exponents = array.exponents();
    let f: A = A::F10[exponents.f as usize];
    let e: A = A::IF10[exponents.e as usize];

    // Execute child and copy to device
    let canonical = array.encoded().clone().execute_cuda(ctx).await?;
    let primitive = canonical.into_primitive();
    let PrimitiveArrayParts {
        buffer, validity, ..
    } = primitive.into_parts();

    let device_input: BufferHandle = if buffer.is_on_device() {
        buffer
    } else {
        ctx.copy_buffer_to_device_async::<A::ALPInt>(buffer)?
            .await?
    };

    // Get CUDA view of input
    let input_view = device_input.cuda_view::<A::ALPInt>()?;

    // Allocate output buffer
    let output_slice = ctx.device_alloc::<A>(array_len)?;
    let output_buf = CudaDeviceBuffer::new(output_slice);
    let output_view = output_buf.as_view();

    let array_len_u64 = array_len as u64;

    // Load kernel function
    let kernel_ptypes = [A::ALPInt::PTYPE, A::PTYPE];
    let cuda_function = ctx.load_function("alp", &kernel_ptypes)?;
    let mut launch_builder = ctx.launch_builder(&cuda_function);

    // Build launch args: input, output, f, e, length
    launch_builder.arg(&input_view);
    launch_builder.arg(&output_view);
    launch_builder.arg(&f);
    launch_builder.arg(&e);
    launch_builder.arg(&array_len_u64);

    // Launch kernel
    let _cuda_events =
        launch_cuda_kernel_impl(&mut launch_builder, CU_EVENT_DISABLE_TIMING, array_len)?;

    // Build result with newly allocated buffer
    let output_handle = BufferHandle::new_device(Arc::new(output_buf));
    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        output_handle,
        A::PTYPE,
        validity,
    )))
}

#[cfg(test)]
#[cfg(cuda_available)]
mod tests {
    use vortex_alp::ALPArray;
    use vortex_alp::Exponents;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity::NonNullable;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_session::VortexSession;

    use super::*;
    use crate::session::CudaSession;

    #[tokio::test]
    async fn test_cuda_alp_decompression_f32() {
        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create encoded values (what ALP would produce)
        // For f32 with exponents (e=0, f=2): decoded = encoded * F10[2] * IF10[0]
        //                                            = encoded * 100.0 * 1.0
        // So encoded value of 100 -> decoded 10000.0
        let encoded_data: Vec<i32> = vec![100, 200, 300, 400, 500];
        let exponents = Exponents { e: 0, f: 2 }; // multiply by 100

        let alp_array = ALPArray::try_new(
            PrimitiveArray::new(Buffer::from(encoded_data.clone()), NonNullable).into_array(),
            exponents,
            None,
        )
        .vortex_expect("failed to create ALP array");

        let result = ALPExecutor
            .execute(alp_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");

        let result_buf =
            Buffer::<f32>::from_byte_buffer(result.as_primitive().buffer_handle().to_host().await);

        assert_eq!(result_buf.len(), encoded_data.len());

        // Check decoded values
        let expected: Vec<f32> = encoded_data.iter().map(|&v| v as f32 * 100.0).collect();
        for (i, (&got, &want)) in result_buf
            .as_slice()
            .iter()
            .zip(expected.iter())
            .enumerate()
        {
            assert!(
                (got - want).abs() < 1e-6,
                "Mismatch at {}: got {}, want {}",
                i,
                got,
                want
            );
        }
    }
}
