// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::sync::Arc;

use async_trait::async_trait;
use cudarc::driver::DeviceRepr;
use cudarc::driver::PushKernelArg;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::buffer::BufferHandle;
use vortex::array::match_each_native_ptype;
use vortex::dtype::NativePType;
use vortex::dtype::Nullability;
use vortex::encodings::sequence::Sequence;
use vortex::encodings::sequence::SequenceDataParts;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

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
            .try_downcast::<Sequence>()
            .map_err(|_| vortex_err!("SequenceExecutor can only accept SequenceArray"))?;

        let len = array.len();
        let nullability = array.dtype().nullability();

        let SequenceDataParts {
            base,
            multiplier,
            ptype,
        } = array.into_data().into_parts();

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

    let kernel_func = ctx.load_function("sequence", &[T::PTYPE])?;

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

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::assert_arrays_eq;
    use vortex::dtype::NativePType;
    use vortex::dtype::Nullability;
    use vortex::encodings::sequence::Sequence;
    use vortex::scalar::PValue;
    use vortex::session::VortexSession;

    use crate::CanonicalCudaExt;
    use crate::CudaSession;
    use crate::executor::CudaExecute;
    use crate::kernel::encodings::sequence::SequenceExecutor;

    #[rstest]
    #[case::u8(10u8, 2u8, 10)]
    #[case::u16(10u16, 2u16, 100)]
    #[case::u32(10u32, 2u32, 1000)]
    #[case::u64(100u64, 20u64, 500)]
    #[crate::test]
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

        let array = Sequence::try_new_typed(base, multiplier, nullability, len).unwrap();

        let cpu_result = crate::canonicalize_cpu(array.clone()).unwrap().into_array();

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
