// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;
use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
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
use vortex_cuda_macros::cuda_tests;
use vortex_dtype::NativePType;
use vortex_dtype::match_each_unsigned_integer_ptype;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::CudaDeviceBuffer;
use crate::arrow::ensure_device_resident;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::kernel::patches::execute_patches;
use crate::kernel::scalar_launch_config;
use crate::launcher::Function;
use crate::launcher::Kernel;
use crate::launcher::Launcher;

struct ALPArgs<T> {
    input: CudaDeviceBuffer,
    output: CudaDeviceBuffer,
    f: T,
    e: T,
    len: u64,
    _marker: std::marker::PhantomData<T>,
}

struct ALPKernel<T> {
    function: Function,
    _marker: std::marker::PhantomData<T>,
}

impl<T: ALPFloat> Kernel for ALPKernel<T> {
    type Args = ALPArgs<T>;

    fn new(function: Function) -> ALPKernel<T> {
        Self {
            function,
            _marker: std::marker::PhantomData,
        }
    }

    unsafe fn launch(self, mut args: ALPArgs<T>, launcher: &Arc<dyn Launcher>) -> VortexResult<()> {
        let len = args.len as usize;
        let args = vec![
            args.input.device_ptr(),
            args.output.device_ptr(),
            std::ptr::addr_of_mut!(args.len).cast(),
        ];

        // SAFETY: pointers are to valid memory at the time of the call.
        unsafe { launcher.launch(self.function, scalar_launch_config(len), args) }
    }
}

/// CUDA decoder for ALP (Adaptive Lossless floating-Point) decompression.
#[derive(Debug)]
pub struct ALPExecutor;

#[async_trait]
impl CudaExecute for ALPExecutor {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let array = array
            .try_into::<ALPVTable>()
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

    let device_input = ensure_device_resident(buffer, ctx).await?;

    // Allocate output buffer
    let output_slice = ctx.device_alloc::<A>(array_len)?;
    let output_buf = CudaDeviceBuffer::new(output_slice);
    let output_view = output_buf.as_view::<A>();

    let array_len_u64 = array_len as u64;

    let module = ctx.load_module("alp")?;
    let alp_kernel = module.load::<ALPKernel<A>>(format!("alp_{}", A::PTYPE))?;

    // wait for launch?
    ctx.launch(
        alp_kernel,
        ALPArgs {
            input: device_input,
            // TODO(aduffy): this is gross. We should make this a futures-based API?
            //  the output_buf is returned back when the kernel completes.
            output: output_buf.clone(),
            f,
            e,
            len: array_len_u64,
            _marker: std::marker::PhantomData,
        },
    )?;

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

#[cuda_tests]
mod tests {
    use vortex_alp::ALPArray;
    use vortex_alp::Exponents;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::patches::Patches;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::buffer;
    use vortex_error::VortexExpect;
    use vortex_session::VortexSession;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    #[tokio::test]
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

        let alp_array = ALPArray::try_new(
            PrimitiveArray::new(Buffer::from(encoded_data.clone()), Validity::NonNullable)
                .into_array(),
            exponents,
            Some(patches),
        )?;

        let cpu_result = alp_array.to_canonical()?.into_array();

        let gpu_result = ALPExecutor
            .execute(alp_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result, gpu_result);

        Ok(())
    }
}
