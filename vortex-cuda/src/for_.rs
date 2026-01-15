// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use vortex_array::Array;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
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
        return Ok(array.to_array().to_canonical());
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
    launch_cuda_kernel!(
        execution_ctx: ctx,
        module: "for",
        ptypes: &[array.ptype()],
        launch_args: [device_data, reference, array_len],
        array_len: array.len()
    );

    let result = ctx.to_host(
        // TODO: Don't copy back after the end of each run.
        &device_data,
        // TODO: Proper alignment
        vortex_buffer::Alignment::of::<P>(),
    )?;

    let primitive = vortex_array::arrays::PrimitiveArray::new(
        result,
        vortex_array::validity::Validity::NonNullable,
    )
    .reinterpret_cast(array.ptype());

    Ok(Canonical::Primitive(primitive))
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_error::VortexExpect;
    use vortex_fastlanes::FoRArray;
    use vortex_session::VortexSession;

    use super::*;
    use crate::has_nvcc;
    use crate::session::CudaSession;

    #[tokio::test]
    async fn test_cuda_for_decompression() {
        if !has_nvcc() {
            return;
        }

        let mut cuda_ctx = CudaSession::new_ctx(VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let for_array = FoRArray::try_new(
            PrimitiveArray::new(
                Buffer::from((0u32..5000).collect::<Vec<u32>>()),
                Validity::NonNullable,
            )
            .into_array(),
            10u32.into(),
        )
        .vortex_expect("failed to create FoR array");

        // Decompress on the GPU.
        let result = execute_for(&for_array, &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed");

        assert_eq!(
            result.as_primitive().as_slice::<u32>(),
            (10u32..5010).collect::<Vec<u32>>()
        );
    }
}
