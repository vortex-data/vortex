// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::primitive::PrimitiveArrayParts;
use vortex::array::buffer::BufferHandle;
use vortex::array::match_each_unsigned_integer_ptype;
use vortex::dtype::NativePType;
use vortex::encodings::alp::ALP;
use vortex::encodings::alp::ALPArray;
use vortex::encodings::alp::ALPFloat;
use vortex::encodings::alp::match_each_alp_float_ptype;
use vortex::error::VortexResult;
use vortex::error::vortex_ensure;
use vortex::error::vortex_err;

use crate::CudaBufferExt;
use crate::CudaDeviceBuffer;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::kernel::patches::execute_patches;

/// CUDA decoder for ALP (Adaptive Lossless floating-Point) decompression.
#[derive(Debug)]
pub(crate) struct ALPExecutor;

#[async_trait]
impl CudaExecute for ALPExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let array = array
            .try_into::<ALP>()
            .map_err(|_| vortex_err!("Expected ALPArray"))?;

        match_each_alp_float_ptype!(array.ptype(), |A| { decode_alp::<A>(array, ctx).await })
    }
}

async fn decode_alp<A>(array: ALPArray, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical>
where
    A: ALPFloat + NativePType + DeviceRepr + Send + Sync + 'static,
    A::ALPInt: NativePType + DeviceRepr + Send + Sync + 'static,
{
    let array_len = array.encoded().len();
    vortex_ensure!(array_len > 0, "ALP array must not be empty");

    // Get the exponent factors from the lookup tables.
    let exponents = array.exponents();
    let f: A = A::F10[exponents.f as usize];
    let e: A = A::IF10[exponents.e as usize];

    // Execute child and copy to device
    let canonical = array.encoded().clone().execute_cuda(ctx).await?;
    let primitive = canonical.into_primitive();
    let PrimitiveArrayParts {
        buffer, validity, ..
    } = primitive.into_data().into_parts();

    let device_input = ctx.ensure_on_device(buffer).await?;

    // Get CUDA view of input
    let input_view = device_input.cuda_view::<A::ALPInt>()?;

    // Allocate output buffer
    let output_slice = ctx.device_alloc::<A>(array_len)?;
    let output_buf = CudaDeviceBuffer::new(output_slice);
    let output_view = output_buf.as_view::<A>();

    let array_len_u64 = array_len as u64;

    // Load kernel function
    let kernel_ptypes = [A::ALPInt::PTYPE, A::PTYPE];
    let cuda_function = ctx.load_function("alp", &kernel_ptypes)?;

    ctx.launch_kernel(&cuda_function, array_len, |args| {
        args.arg(&input_view)
            .arg(&output_view)
            .arg(&f)
            .arg(&e)
            .arg(&array_len_u64);
    })?;

    // Check if there are any patches to decode here
    let output_buf = if let Some(patches) = array.patches() {
        match_each_unsigned_integer_ptype!(patches.indices_ptype()?, |I| {
            execute_patches::<A, I>(patches.clone(), output_buf, ctx).await?
        })
    } else {
        output_buf
    };

    // TODO(aduffy): scatter patch values validity. There are several places we'll need to start
    //  handling validity.

    let output_handle = BufferHandle::new_device(Arc::new(output_buf));
    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        output_handle,
        A::PTYPE,
        validity,
    )))
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::patches::Patches;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::buffer::buffer;
    use vortex::encodings::alp::ALP;
    use vortex::encodings::alp::Exponents;
    use vortex::error::VortexExpect;
    use vortex::session::VortexSession;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    #[crate::test]
    async fn test_cuda_alp_decompression_f32() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Create encoded values (what ALP would produce)
        // For f32 with exponents (e=0, f=2): decoded = encoded * F10[2] * IF10[0]
        //                                            = encoded * 100.0 * 1.0
        // So encoded value of 100 -> decoded 10000.0
        let encoded_data: Vec<i32> = vec![100, 200, 300, 400, 500];
        let exponents = Exponents { e: 0, f: 2 }; // multiply by 100

        // Patches
        let patches = Patches::new(
            5,
            0,
            PrimitiveArray::new(buffer![0u32, 4u32], Validity::NonNullable).into_array(),
            PrimitiveArray::new(buffer![0.0f32, 999f32], Validity::NonNullable).into_array(),
            None,
        )
        .unwrap();

        let alp_array = ALP::try_new(
            PrimitiveArray::new(Buffer::from(encoded_data.clone()), Validity::NonNullable)
                .into_array(),
            exponents,
            Some(patches),
        )?;

        let cpu_result = alp_array.to_canonical()?.into_array();

        let gpu_result = ALPExecutor
            .execute(alp_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result, gpu_result);

        Ok(())
    }
}
