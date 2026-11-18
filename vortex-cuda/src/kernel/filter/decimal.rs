// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::Canonical;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::decimal::DecimalDataParts;
use vortex::dtype::NativeDecimalType;
use vortex::error::VortexResult;
use vortex::mask::Mask;
use vortex_cub::filter::CubFilterable;
use vortex_cuda_driver::DeviceRepr;

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
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::arrays::DecimalArray;
    use vortex::array::arrays::FilterArray;
    use vortex::array::assert_arrays_eq;
    use vortex::dtype::DecimalDType;
    use vortex::dtype::DecimalType;
    use vortex::dtype::i256;
    use vortex::error::VortexResult;
    use vortex::mask::Mask;
    use vortex::session::VortexSession;
    use vortex_array::VortexSessionExecute;

    use crate::CanonicalCudaExt;
    use crate::FilterExecutor;
    use crate::executor::CudaExecute;
    use crate::session::CudaSession;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case::i32_sparse(
        DecimalType::I32,
        DecimalArray::from_iter([1i32, 2, 3, 4, 5, 6, 7, 8], DecimalDType::new(9, 5)),
        Mask::from_iter([true, false, true, false, true, false, true, false])
    )]
    #[case::i32_dense(
        DecimalType::I32,
        DecimalArray::from_iter([10i32, 20, 30, 40, 50], DecimalDType::new(9, 5)),
        Mask::from_iter([true, true, true, false, true])
    )]
    #[case::i64_large(
        DecimalType::I64,
        DecimalArray::from_iter(0..1000i64, DecimalDType::new(18, 5)),
        Mask::from_iter((0..1000).map(|i| i % 3 == 0))
    )]
    #[case::i8_all_true(
        DecimalType::I8,
        DecimalArray::from_iter([1i8, 2, 3, 4, 5], DecimalDType::new(2, 1)),
        Mask::from_iter([true, true, true, true, true])
    )]
    #[case::i8_nullable(
        DecimalType::I8,
        DecimalArray::from_option_iter([Some(-3i8), None, Some(8)], DecimalDType::new(2, 1)),
        Mask::from_iter([true, true, false])
    )]
    #[case::i16_values(
        DecimalType::I16,
        DecimalArray::from_iter([-700i16, 100, 900], DecimalDType::new(4, 2)),
        Mask::from_iter([true, false, true])
    )]
    #[case::i32_all_false(
        DecimalType::I32,
        DecimalArray::from_iter([1i32, 2, 3, 4, 5], DecimalDType::new(9, 5)),
        Mask::from_iter([false, false, false, false, false])
    )]
    #[case::i128_values(
        DecimalType::I128,
        DecimalArray::from_iter([1i128, 2, 3, 4, 5], DecimalDType::new(19, 5)),
        Mask::from_iter([false, true, false, true, false])
    )]
    #[case::i256_values(
        DecimalType::I256,
        DecimalArray::from_iter([1, 2, 3, 4, 5].map(i256::from_i128), DecimalDType::new(76, 5)),
        Mask::from_iter([false, true, false, true, false])
    )]
    #[case::i32_multiblock(
        DecimalType::I32,
        DecimalArray::from_iter(0..2050i32, DecimalDType::new(9, 5)),
        Mask::from_iter((0..2050).map(|i| i % 7 == 0))
    )]
    #[crate::test]
    async fn test_gpu_filter_decimal(
        #[case] values_type: DecimalType,
        #[case] input: DecimalArray,
        #[case] mask: Mask,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())?;
        assert_eq!(input.values_type(), values_type);
        let filter_array = FilterArray::try_new(input.into_array(), mask)?;
        let gpu_result = FilterExecutor
            .execute(filter_array.clone().into_array(), &mut cuda_ctx)
            .await?
            .into_host(cuda_ctx.execution_ctx())
            .await?
            .into_decimal();

        assert_eq!(gpu_result.values_type(), values_type);
        assert_arrays_eq!(gpu_result.into_array(), filter_array, &mut ctx);

        Ok(())
    }
}
