// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::match_each_unsigned_integer_ptype;
use vortex::dtype::NativePType;
use vortex::dtype::PType;
use vortex::encodings::zigzag::ZigZag;
use vortex::encodings::zigzag::ZigZagArray;
use vortex::encodings::zigzag::ZigZagArrayExt;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;

use crate::CudaBufferExt;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;

/// CUDA decoder for ZigZag decoding.
#[derive(Debug)]
pub(crate) struct ZigZagExecutor;

impl ZigZagExecutor {
    fn try_specialize(array: ArrayRef) -> Option<ZigZagArray> {
        array.try_downcast::<ZigZag>().ok()
    }
}

#[async_trait]
impl CudaExecute for ZigZagExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
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
    vortex_ensure!(array_len > 0, "ZigZag array must not be empty");

    // Execute child and copy to device
    let canonical = array.encoded().clone().execute_cuda(ctx).await?;
    let primitive = canonical.into_primitive();
    let PrimitiveDataParts {
        buffer, validity, ..
    } = primitive.into_data_parts();

    let device_buffer = ctx.ensure_on_device(buffer).await?;

    // Get CUDA view of the buffer
    let cuda_view = device_buffer.cuda_view::<U>()?;
    let array_len_u64 = array_len as u64;

    // Load kernel function
    let cuda_function = ctx.load_function("zigzag", &[U::PTYPE])?;

    ctx.launch_kernel(&cuda_function, array_len, |args| {
        args.arg(&cuda_view).arg(&array_len_u64);
    })?;

    // Build result - in-place, reinterpret as signed
    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        device_buffer,
        output_ptype,
        validity,
    )))
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::validity::Validity::NonNullable;
    use vortex::buffer::Buffer;
    use vortex::encodings::zigzag::ZigZag;
    use vortex::error::VortexExpect;
    use vortex::session::VortexSession;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    #[crate::test]
    async fn test_cuda_zigzag_decompression_u32() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // ZigZag encoding: 0->0, 1->-1, 2->1, 3->-2, 4->2, ...
        // So encoded [0, 2, 4, 1, 3] should decode to [0, 1, 2, -1, -2]
        let encoded_data: Vec<u32> = vec![0, 2, 4, 1, 3];

        let zigzag_array = ZigZag::try_new(
            PrimitiveArray::new(Buffer::from(encoded_data), NonNullable).into_array(),
        )?;

        let cpu_result = crate::canonicalize_cpu(zigzag_array.clone())?;

        let gpu_result = ZigZagExecutor
            .execute(zigzag_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }
}
