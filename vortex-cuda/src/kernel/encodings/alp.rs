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
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::array::buffer::BufferHandle;
use vortex::array::match_each_unsigned_integer_ptype;
use vortex::dtype::NativePType;
use vortex::encodings::alp::ALP;
use vortex::encodings::alp::ALPArray;
use vortex::encodings::alp::ALPArrayExt;
use vortex::encodings::alp::ALPArraySlotsExt;
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
            .try_downcast::<ALP>()
            .map_err(|_| vortex_err!("Expected ALPArray"))?;

        match_each_alp_float_ptype!(array.dtype().as_ptype(), |A| {
            decode_alp::<A>(array, ctx).await
        })
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
    let PrimitiveDataParts {
        buffer, validity, ..
    } = primitive.into_data_parts();

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

    // Check if there are any patches to decode here. Patch validity does not
    // need to be scattered: the ALP encoder strips null positions from the
    // exception list, so patches only exist at valid positions. execute_patches
    // additionally guards against nullable patch values at runtime.
    let output_buf = if let Some(patches) = array.patches() {
        match_each_unsigned_integer_ptype!(patches.indices_ptype()?, |I| {
            execute_patches::<A, I>(patches.clone(), output_buf, ctx).await?
        })
    } else {
        output_buf
    };

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
    use vortex::encodings::alp::alp_encode;
    use vortex::error::VortexExpect;
    use vortex::session::VortexSession;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::executor::CudaArrayExt;
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

    /// ALP with nullable encoded data and patches — the encoder strips null
    /// positions from the exception list, so patch validity doesn't need
    /// scattering. This test verifies that the encoded child's validity is
    /// preserved through the standalone ALP GPU executor.
    #[crate::test]
    async fn test_cuda_alp_nullable_with_patches() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        // Values that will produce ALP exceptions at non-null positions.
        // Nulls at positions 1 and 3; the exception at position 4 (1.23456)
        // can't be encoded losslessly by ALP.
        let values: Vec<Option<f32>> = vec![
            Some(1.0),
            None,
            Some(2.0),
            None,
            Some(1.23456),
            Some(3.0),
            Some(4.0),
            Some(5.0),
        ];
        let prim = PrimitiveArray::from_option_iter(values);
        let alp_array = alp_encode(&prim, None)?;

        let cpu_result = alp_array.to_canonical()?.into_array();

        let gpu_result = alp_array
            .into_array()
            .execute_cuda(&mut cuda_ctx)
            .await?
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result, gpu_result);
        Ok(())
    }

    /// ALP with all-valid nullable data — the dtype is nullable but no
    /// elements are actually null.
    #[crate::test]
    async fn test_cuda_alp_all_valid_nullable() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create execution context");

        let values = PrimitiveArray::new(
            Buffer::from(vec![1.0f32, 2.0, 3.0, 4.0, 5.0]),
            Validity::AllValid,
        );
        let alp_array = alp_encode(&values, None)?;

        let cpu_result = alp_array.to_canonical()?.into_array();

        let gpu_result = alp_array
            .into_array()
            .execute_cuda(&mut cuda_ctx)
            .await?
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result, gpu_result);
        Ok(())
    }
}
