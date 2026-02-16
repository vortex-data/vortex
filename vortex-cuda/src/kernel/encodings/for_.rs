// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::PrimitiveArrayParts;
use vortex_cuda_macros::cuda_tests;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_native_simd_ptype;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_error::vortex_ensure;
use vortex_error::vortex_err;
use vortex_fastlanes::FoRArray;
use vortex_fastlanes::FoRVTable;

use crate::CudaBufferExt;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;

/// CUDA decoder for frame-of-reference.
#[derive(Debug)]
pub(crate) struct FoRExecutor;

impl FoRExecutor {
    fn try_specialize(array: ArrayRef) -> Option<FoRArray> {
        array.try_into::<FoRVTable>().ok()
    }
}

#[async_trait]
impl CudaExecute for FoRExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
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
    vortex_ensure!(array_len > 0, "FoR encoded array must not be empty");

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

    let device_buffer = ctx.ensure_on_device(buffer).await?;

    // Get CUDA view of the buffer
    let cuda_view = device_buffer.cuda_view::<P>()?;
    let array_len_u64 = array_len as u64;

    // Load kernel function
    let kernel_ptypes = [P::PTYPE];
    let cuda_function = ctx.load_function_ptype("for", &kernel_ptypes)?;

    ctx.launch_kernel(&cuda_function, array_len, |args| {
        args.arg(&cuda_view).arg(&reference).arg(&array_len_u64);
    })?;

    // Build result - in-place reuses the same buffer
    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        device_buffer,
        P::PTYPE,
        validity,
    )))
}

#[cuda_tests]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity::NonNullable;
    use vortex_buffer::Buffer;
    use vortex_dtype::NativePType;
    use vortex_error::VortexExpect;
    use vortex_fastlanes::FoRArray;
    use vortex_scalar::Scalar;
    use vortex_session::VortexSession;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    fn make_for_array<T: NativePType + Into<Scalar>>(input_data: Vec<T>, reference: T) -> FoRArray {
        FoRArray::try_new(
            PrimitiveArray::new(Buffer::from(input_data), NonNullable).into_array(),
            reference.into(),
        )
        .unwrap()
    }

    #[rstest]
    #[case::u8(make_for_array((0..2050).map(|i| (i % 246) as u8).collect(), 10u8))]
    #[case::u16(make_for_array((0..2050).map(|i| (i % 2050) as u16).collect(), 1000u16))]
    #[case::u32(make_for_array((0..2050).map(|i| (i % 2050) as u32).collect(), 100000u32))]
    #[case::u64(make_for_array((0..2050).map(|i| (i % 2050) as u64).collect(), 1000000u64))]
    #[tokio::test]
    async fn test_cuda_for_decompression(#[case] for_array: FoRArray) -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let cpu_result = for_array.to_canonical()?;

        let gpu_result = FoRExecutor
            .execute(for_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result.into_array(), gpu_result);

        Ok(())
    }
}
