// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use cudarc::driver::DeviceRepr;
use vortex::array::Canonical;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::decimal::DecimalDataParts;
use vortex::dtype::NativeDecimalType;
use vortex::error::VortexResult;
use vortex::mask::Mask;
use vortex_cub::filter::CubFilterable;

use crate::CudaExecutionCtx;
use crate::kernel::filter::filter_sized;

pub(super) async fn filter_decimal<D: NativeDecimalType + DeviceRepr + CubFilterable>(
    array: DecimalArray,
    mask: Mask,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let DecimalDataParts {
        values,
        validity,
        decimal_dtype,
        ..
    } = array.into_data_parts();

    let filtered_validity = validity.filter(&mask)?;
    let filtered_values = filter_sized::<D>(values, mask, ctx).await?;

    Ok(Canonical::Decimal(DecimalArray::new_handle(
        filtered_values,
        D::DECIMAL_TYPE,
        decimal_dtype,
        filtered_validity,
    )))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::arrays::DecimalArray;
    use vortex::array::arrays::FilterArray;
    use vortex::array::assert_arrays_eq;
    use vortex::dtype::DecimalDType;
    use vortex::dtype::i256;
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
        DecimalArray::from_iter([1i32, 2, 3, 4, 5, 6, 7, 8], DecimalDType::new(19, 5)),
        Mask::from_iter([true, false, true, false, true, false, true, false])
    )]
    #[case::i32_dense(
        DecimalArray::from_iter([10i32, 20, 30, 40, 50], DecimalDType::new(19, 5)),
        Mask::from_iter([true, true, true, false, true])
    )]
    #[case::i64_large(
        DecimalArray::from_iter(0..1000i64, DecimalDType::new(19, 5)),
        Mask::from_iter((0..1000).map(|i| i % 3 == 0))
    )]
    #[case::i8_all_true(
        DecimalArray::from_iter([1i8, 2, 3, 4, 5], DecimalDType::new(19, 5)),
        Mask::from_iter([true, true, true, true, true])
    )]
    #[case::i32_all_false(
        DecimalArray::from_iter([1i32, 2, 3, 4, 5], DecimalDType::new(19, 5)),
        Mask::from_iter([false, false, false, false, false])
    )]
    #[case::i128_values(
        DecimalArray::from_iter([1i128, 2, 3, 4, 5], DecimalDType::new(19, 5)),
        Mask::from_iter([false, true, false, true, false])
    )]
    #[case::i256_values(
        DecimalArray::from_iter([i256::from_i128(1), i256::from_i128(2), i256::from_i128(3), i256::from_i128(4), i256::from_i128(5)], DecimalDType::new(19, 5)),
        Mask::from_iter([false, true, false, true, false])
    )]
    #[crate::test]
    async fn test_gpu_filter_decimal(
        #[case] input: DecimalArray,
        #[case] mask: Mask,
    ) -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create CUDA execution context");

        let filter_array = FilterArray::try_new(input.clone().into_array(), mask.clone())?;

        let cpu_result = crate::canonicalize_cpu(filter_array.clone())?.into_array();

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
    async fn test_gpu_filter_decimal_large_array() -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create CUDA execution context");

        // Create a large array to test multi-block execution
        let data: Vec<i32> = (0..2050).collect();
        let input = DecimalArray::from_iter(data, DecimalDType::new(19, 5));

        // Select every 7th element
        let mask = Mask::from_iter((0..2050).map(|i| i % 7 == 0));

        let filter_array = FilterArray::try_new(input.into_array(), mask)?;

        let cpu_result = crate::canonicalize_cpu(filter_array.clone())?.into_array();

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
