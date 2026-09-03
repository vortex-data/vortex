// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::Masked;
use vortex::array::arrays::masked::MaskedArrayExt;
use vortex::array::arrays::masked::MaskedArraySlotsExt;
use vortex::array::arrays::masked::mask_validity_canonical;
use vortex::array::validity::Validity;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::executor::execute_validity_cuda;

/// CUDA executor for MaskedArray.
///
/// A `MaskedArray` is a child array that carries no nulls of its own, plus the validity
/// bitmap that supplies them. Decode the child on the GPU, decode the mask on the GPU, and
/// attach the mask to the result.
#[derive(Debug)]
pub(crate) struct MaskedExecutor;

#[async_trait]
impl CudaExecute for MaskedExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let masked = array
            .try_downcast::<Masked>()
            .map_err(|_| vortex_err!("Expected MaskedArray"))?;

        let len = masked.len();
        let validity = masked.masked_validity();

        // `MaskedArray` guarantees its child holds no nulls, so the mask alone determines the
        // output validity. Combining two device-resident bitmaps would need a CPU compute pass.
        if matches!(masked.child().validity()?, Validity::Array(_)) {
            vortex_bail!(
                "MaskedArray child carries a per-element validity bitmap, which cannot be combined with the mask on the GPU"
            );
        }

        let child = masked.child().clone().execute_cuda(ctx).await?;

        let validity = execute_validity_cuda(validity, len, ctx).await?;
        mask_validity_canonical(child, validity, ctx.execution_ctx())
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray;
    use vortex::array::arrays::BoolArray;
    use vortex::array::arrays::MaskedArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::assert_arrays_eq;
    use vortex::buffer::buffer;
    use vortex::error::VortexExpect;
    use vortex_array::VortexSessionExecute;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    #[crate::test]
    async fn test_cuda_masked_applies_validity() -> VortexResult<()> {
        let mut ctx = vortex_array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");

        let child = PrimitiveArray::new(buffer![1i32, 2, 3, 4], Validity::NonNullable).into_array();
        let validity = Validity::Array(
            BoolArray::from_iter([true, false, true, true].into_iter()).into_array(),
        );
        let masked = MaskedArray::try_new(child, validity)?;

        let gpu_result = MaskedExecutor
            .execute(masked.clone().into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decompression failed")
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(masked, gpu_result, &mut ctx);

        Ok(())
    }
}
