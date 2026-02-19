// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::buffer::BufferHandle;
use vortex_array::dtype::NativePType;
use vortex_array::dtype::Nullability;
use vortex_array::match_each_native_ptype;
use vortex_cuda_macros::cuda_tests;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_sequence::SequenceArrayParts;
use vortex_sequence::SequenceVTable;

use crate::CudaDeviceBuffer;
use crate::CudaExecutionCtx;
use crate::executor::CudaExecute;

/// CUDA execution for `SequenceArray`.
#[derive(Debug)]
pub(crate) struct SequenceExecutor;

#[async_trait]
impl CudaExecute for SequenceExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let array = array
            .try_into::<SequenceVTable>()
            .map_err(|_| vortex_err!("SequenceExecutor can only accept SequenceArray"))?;

        let SequenceArrayParts {
            base,
            multiplier,
            len,
            ptype,
            nullability,
        } = array.into_parts();

        match_each_native_ptype!(ptype, |P| {
            let base = base.cast::<P>()?;
            let multiplier = multiplier.cast::<P>()?;
            execute_typed::<P>(base, multiplier, len, nullability, ctx).await
        })
    }
}

async fn execute_typed<T: NativePType + DeviceRepr>(
    base: T,
    multiplier: T,
    len: usize,
    nullability: Nullability,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let buffer = ctx.device_alloc::<T>(len)?;

    let len_u64 = len as u64;

    let kernel_func = ctx.load_function_ptype("sequence", &[T::PTYPE])?;

    ctx.launch_kernel(&kernel_func, len, |args| {
        args.arg(&buffer).arg(&base).arg(&multiplier).arg(&len_u64);
    })?;

    let output_buf = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(buffer)));

    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        output_buf,
        T::PTYPE,
        nullability.into(),
    )))
}

#[cuda_tests]
mod tests {
    use futures::executor::block_on;
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::NativePType;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::PValue;
    use vortex_sequence::SequenceArray;
    use vortex_session::VortexSession;

    use crate::CanonicalCudaExt;
    use crate::CudaSession;
    use crate::executor::CudaExecute;
    use crate::kernel::encodings::sequence::SequenceExecutor;

    #[rstest]
    #[case::u8(10u8, 2u8, 10)]
    #[case::u16(10u16, 2u16, 100)]
    #[case::u32(10u32, 2u32, 1000)]
    #[case::u64(100u64, 20u64, 500)]
    fn test_sequence<T: NativePType + Into<PValue>>(
        #[case] base: T,
        #[case] multiplier: T,
        #[case] len: usize,
    ) {
        block_on(
            async move { test_ptype::<T>(base, multiplier, len, Nullability::NonNullable).await },
        );

        block_on(
            async move { test_ptype::<T>(base, multiplier, len, Nullability::Nullable).await },
        );
    }

    async fn test_ptype<P: NativePType + Into<PValue>>(
        base: P,
        multiplier: P,
        len: usize,
        nullability: Nullability,
    ) {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty()).unwrap();

        let array = SequenceArray::typed_new(base, multiplier, nullability, len).unwrap();

        let cpu_result = array.to_canonical().unwrap().into_array();

        let gpu_result = SequenceExecutor
            .execute(array.into_array(), &mut cuda_ctx)
            .await
            .unwrap()
            .into_host()
            .await
            .unwrap()
            .into_array();

        assert_arrays_eq!(cpu_result, gpu_result);
    }
}
