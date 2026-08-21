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
use vortex::array::match_each_unsigned_integer_ptype;
use vortex::dtype::NativePType;
use vortex::dtype::Nullability;
use vortex::dtype::PType;
use vortex::encodings::sequence::Sequence;
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
    // The kernel intentionally truncates both operands to the output width.
    #[expect(clippy::cast_possible_truncation)]
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
        let output_ptype = PType::try_from(array.dtype())?;

        // Unsigned wrapping arithmetic handles every signedness pair at the output width.
        let (base, multiplier) = array.wrapping_bits()?;

        match_each_unsigned_integer_ptype!(output_ptype.to_unsigned(), |U| {
            execute_typed::<U>(
                base as U,
                multiplier as U,
                len,
                output_ptype,
                nullability,
                ctx,
            )
            .await
        })
    }
}

async fn execute_typed<T: NativePType + DeviceRepr>(
    base: T,
    multiplier: T,
    len: usize,
    output_ptype: PType,
    nullability: Nullability,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let mut buffer = ctx.device_alloc::<T>(len)?;

    let len_u64 = len as u64;

    let kernel_func = ctx.load_function("sequence", &[T::PTYPE])?;

    ctx.launch_kernel(&kernel_func, len, |args| {
        args.arg(&mut buffer)
            .arg(&base)
            .arg(&multiplier)
            .arg(&len_u64);
    })?;

    let output_buf = BufferHandle::new_device(Arc::new(CudaDeviceBuffer::new(buffer)));

    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        output_buf,
        output_ptype,
        nullability.into(),
    )))
}

#[cfg(test)]
mod tests {
    use futures::executor::block_on;
    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::builtins::ArrayBuiltins;
    use vortex::dtype::DType;
    use vortex::dtype::NativePType;
    use vortex::dtype::Nullability;
    use vortex::dtype::PType;
    use vortex::encodings::sequence::Sequence;
    use vortex::error::VortexResult;
    use vortex::scalar::PValue;
    use vortex_array::VortexSessionExecute;

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
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session()).unwrap();

        let array = Sequence::try_new_typed(base, multiplier, nullability, len)
            .unwrap()
            .into_array();

        let gpu_result = SequenceExecutor
            .execute(array.clone(), &mut cuda_ctx)
            .await
            .unwrap()
            .into_host()
            .await
            .unwrap()
            .into_array();

        assert_arrays_eq!(array, gpu_result, &mut ctx);
    }

    #[crate::test]
    async fn test_sequence_arithmetic_ptype_differs_from_output() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session()).unwrap();

        let array = Sequence::try_new_typed(100i32, -10i32, Nullability::NonNullable, 5)?
            .into_array()
            .cast(DType::Primitive(PType::U8, Nullability::NonNullable))?;

        let gpu_result = SequenceExecutor
            .execute(array.clone(), &mut cuda_ctx)
            .await?
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(array, gpu_result, &mut ctx);

        Ok(())
    }
}
