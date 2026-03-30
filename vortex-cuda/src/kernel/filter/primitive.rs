// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use cudarc::driver::DeviceRepr;
use vortex::array::Canonical;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::primitive::PrimitiveArrayParts;
use vortex::dtype::NativePType;
use vortex::error::VortexResult;
use vortex::mask::Mask;
use vortex_cub::filter::CubFilterable;

use crate::CudaExecutionCtx;
use crate::kernel::filter::filter_sized;

/// Execute a filter operation over the primitive array on a GPU.
pub(super) async fn filter_primitive<T>(
    array: PrimitiveArray,
    mask: Mask,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical>
where
    T: NativePType + DeviceRepr + CubFilterable + Send + Sync + 'static,
{
    let PrimitiveArrayParts {
        buffer, validity, ..
    } = array.into_data().into_parts();

    let filtered_validity = validity.filter(&mask)?;
    let filtered_values = filter_sized::<T>(buffer, mask, ctx).await?;

    Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
        filtered_values,
        T::PTYPE,
        filtered_validity,
    )))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::arrays::FilterArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::assert_arrays_eq;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;
    use vortex::mask::Mask;
    use vortex::session::VortexSession;

    use crate::CanonicalCudaExt;
    use crate::FilterExecutor;
    use crate::executor::CudaExecute;
    use crate::session::CudaSession;

    #[rstest]
    #[case::i32_sparse(
        PrimitiveArray::from_iter([1i32, 2, 3, 4, 5, 6, 7, 8]),
        Mask::from_iter([true, false, true, false, true, false, true, false])
    )]
    #[case::i32_dense(
        PrimitiveArray::from_iter([10i32, 20, 30, 40, 50]),
        Mask::from_iter([true, true, true, false, true])
    )]
    #[case::i64_large(
        PrimitiveArray::from_iter((0..1000i64).collect::<Vec<_>>()),
        Mask::from_iter((0..1000).map(|i| i % 3 == 0))
    )]
    #[case::f64_values(
        PrimitiveArray::from_iter([1.1f64, 2.2, 3.3, 4.4, 5.5]),
        Mask::from_iter([false, true, false, true, false])
    )]
    #[case::u8_all_true(
        PrimitiveArray::from_iter([1u8, 2, 3, 4, 5]),
        Mask::from_iter([true, true, true, true, true])
    )]
    #[case::u32_all_false(
        PrimitiveArray::from_iter([1u32, 2, 3, 4, 5]),
        Mask::from_iter([false, false, false, false, false])
    )]
    #[crate::test]
    async fn test_gpu_filter(
        #[case] input: PrimitiveArray,
        #[case] mask: Mask,
    ) -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create CUDA execution context");

        let filter_array = FilterArray::try_new(input.clone().into_array(), mask.clone())?;

        let cpu_result = filter_array.to_canonical()?.into_array();

        let gpu_result = FilterExecutor
            .execute(filter_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU filter failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(cpu_result, gpu_result);

        Ok(())
    }

    #[crate::test]
    async fn test_gpu_filter_large_array() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create CUDA execution context");

        // Create a large array to test multi-block execution
        let data: Vec<i32> = (0..2050).collect();
        let input = PrimitiveArray::from_iter(data);

        // Select every 7th element
        let mask = Mask::from_iter((0..2050).map(|i| i % 7 == 0));

        let filter_array = FilterArray::try_new(input.into_array(), mask)?;

        let cpu_result = filter_array.to_canonical()?.into_array();

        let gpu_result = FilterExecutor
            .execute(filter_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU filter failed")
            .into_host()
            .await?
            .into_array();

        assert_eq!(cpu_result.len(), gpu_result.len());
        assert_arrays_eq!(cpu_result, gpu_result);

        Ok(())
    }
}
