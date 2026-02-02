// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex_array::Canonical;
use vortex_array::arrays::VarBinViewArray;
use vortex_array::arrays::VarBinViewArrayParts;
use vortex_cuda_macros::cuda_tests;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::CudaExecutionCtx;
use crate::kernel::filter::filter_sized;

pub(super) async fn filter_varbinview(
    array: VarBinViewArray,
    mask: Mask,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let VarBinViewArrayParts {
        views,
        buffers,
        validity,
        dtype,
    } = array.into_parts();

    let filtered_validity = validity.filter(&mask)?;

    let d_views = if views.is_on_device() {
        views
    } else {
        ctx.move_to_device(views)?.await?
    };

    let filtered_views = filter_sized::<i128>(d_views, mask, ctx).await?;

    Ok(Canonical::VarBinView(VarBinViewArray::new_handle(
        filtered_views,
        buffers,
        dtype,
        filtered_validity,
    )))
}

#[cuda_tests]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::FilterArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexExpect;
    use vortex_error::VortexResult;
    use vortex_mask::Mask;
    use vortex_session::VortexSession;

    use crate::CanonicalCudaExt;
    use crate::FilterExecutor;
    use crate::executor::CudaExecute;
    use crate::session::CudaSession;

    #[rstest]
    #[case::nato(
        VarBinViewArray::from_iter_str(["alpha", "bravo", "charlie", "delta"]),
        Mask::from_iter([true, false, true, false])
    )]
    #[case::planets(
        VarBinViewArray::from_iter_str(
            ["mercury", "venus", "earth", "mars", "jupiter", "saturn", "uranus", "neptune", "pluto"]
        ),
        Mask::from_iter([true, true, true, true, true, true, true, true, false])
    )]
    #[tokio::test]
    async fn test_gpu_filter_strings(
        #[case] input: VarBinViewArray,
        #[case] mask: Mask,
    ) -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("failed to create CUDA execution context");

        let filter_array = FilterArray::try_new(input.into_array(), mask.clone())?;

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
}
