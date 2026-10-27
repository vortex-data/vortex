// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use vortex::array::Canonical;
use vortex::array::arrays::VarBinViewArray;
use vortex::array::arrays::varbinview::VarBinViewDataParts;
use vortex::error::VortexResult;
use vortex::mask::Mask;

use crate::CudaExecutionCtx;
use crate::kernel::filter::filter_sized;

pub(super) async fn filter_varbinview(
    array: VarBinViewArray,
    mask: Mask,
    ctx: &mut CudaExecutionCtx,
) -> VortexResult<Canonical> {
    let VarBinViewDataParts {
        views,
        buffers,
        validity,
        dtype,
    } = array.into_data_parts();

    let filtered_validity = validity.filter(&mask)?;

    let d_views = ctx.ensure_on_device(views).await?;

    let filtered_views = filter_sized::<i128>(d_views, mask, ctx).await?;

    Ok(Canonical::VarBinView(VarBinViewArray::new_handle(
        filtered_views,
        buffers,
        dtype,
        filtered_validity,
    )))
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::arrays::FilterArray;
    use vortex::array::arrays::VarBinViewArray;
    use vortex::array::assert_arrays_eq;
    use vortex::error::VortexExpect;
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
    #[crate::test]
    async fn test_gpu_filter_strings(
        #[case] input: VarBinViewArray,
        #[case] mask: Mask,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create CUDA execution context");

        let filter_array = FilterArray::try_new(input.into_array(), mask.clone())?;

        let gpu_result = FilterExecutor
            .execute(filter_array.clone().into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU filter failed")
            .into_host(cuda_ctx.execution_ctx())
            .await?
            .into_array();

        assert_arrays_eq!(filter_array, gpu_result, &mut ctx);

        Ok(())
    }
}
