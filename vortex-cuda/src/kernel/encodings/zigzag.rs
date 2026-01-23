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
use vortex_dtype::PType;
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_zigzag::ZigZagArray;
use vortex_zigzag::ZigZagVTable;

use crate::CudaBufferExt;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::launch_cuda_kernel_impl;

/// CUDA decoder for ZigZag decoding.
#[derive(Debug)]
pub struct ZigZagExecutor;

impl ZigZagExecutor {
    fn try_specialize(array: ArrayRef) -> Option<ZigZagArray> {
        array.try_into::<ZigZagVTable>().ok()
    }
}

#[async_trait]
impl CudaExecute for ZigZagExecutor {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let array =
            Self::try_specialize(array).ok_or_else(|| vortex_err!("Expected ZigZagArray"))?;

        // The encoded array is unsigned, we decode to signed of the same width.
        let encoded_ptype = array.encoded().dtype().as_ptype();
        let output_ptype = PType::try_from(array.dtype())?;

        match_each_unsigned_integer_ptype!(encoded_ptype, |U| {
            decode_zigzag::<U>(array, output_ptype, ctx).await
        })
    }
}

async fn decode_zigzag<U>(
    array: ZigZagArray,
    output_ptype: PType,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical>
where
    U: NativePType + DeviceRepr + Send + Sync + 'static,
{
    let array_len = array.encoded().len();
    assert!(array_len > 0);

    // Execute child and copy to device
    let canonical = array.encoded().clone().execute_cuda(ctx).await?;
    let primitive = canonical.into_primitive();
    let PrimitiveArrayParts {
        buffer, validity, ..
    } = primitive.into_parts();

    let device_buffer: BufferHandle = if buffer.is_on_device() {
        buffer
    } else {
        ctx.move_to_device::<U>(buffer)?.await?
    };

    // Get CUDA view of the buffer
    let cuda_view = device_buffer.cuda_view::<U>()?;
    let array_len_u64 = array_len as u64;

    // Load kernel function
    let kernel_ptypes = [U::PTYPE];
    let cuda_function = ctx.load_function_ptype("zigzag", &kernel_ptypes)?;
    let mut launch_builder = ctx.launch_builder(&cuda_function);

    // Build launch args: buffer, length
    launch_builder.arg(&cuda_view);
    launch_builder.arg(&array_len_u64);

    // Launch kernel
    let _cuda_events =
        launch_cuda_kernel_impl(&mut launch_builder, CU_EVENT_DISABLE_TIMING, array_len)?;

    // Build result - in-place, reinterpret as signed
    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        device_buffer,
        output_ptype,
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
    use vortex_session::VortexSession;
    use vortex_zigzag::ZigZagArray;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    #[tokio::test]
    async fn test_cuda_zigzag_decompression_u32() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // ZigZag encoding: 0->0, 1->-1, 2->1, 3->-2, 4->2, ...
        // So encoded [0, 2, 4, 1, 3] should decode to [0, 1, 2, -1, -2]
        let encoded_data: Vec<u32> = vec![0, 2, 4, 1, 3];

        let zigzag_array = ZigZagArray::try_new(
            PrimitiveArray::new(Buffer::from(encoded_data), NonNullable).into_array(),
        )?;

        let cpu_result = zigzag_array.to_canonical()?;

        let gpu_result = ZigZagExecutor
            .execute(zigzag_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .to_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }
}
