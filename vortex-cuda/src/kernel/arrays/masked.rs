// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::Masked;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::masked::MaskedArrayExt;
use vortex::array::arrays::masked::MaskedArraySlotsExt;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_err;

use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;
use crate::executor::execute_validity_cuda;

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
        let array = array
            .try_downcast::<Masked>()
            .map_err(|_| vortex_err!("Expected MaskedArray"))?;
        let len = array.len();
        let validity = execute_validity_cuda(array.masked_validity(), len, ctx).await?;

        match array.child().clone().execute_cuda(ctx).await? {
            Canonical::Primitive(primitive) => {
                let PrimitiveDataParts { ptype, buffer, .. } = primitive.into_data_parts();
                Ok(Canonical::Primitive(PrimitiveArray::from_buffer_handle(
                    buffer, ptype, validity,
                )))
            }
            canonical => vortex_bail!(
                "CUDA Masked execution currently supports primitive children, got {}",
                canonical.dtype()
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use vortex::array::IntoArray;
    use vortex::array::VortexSessionExecute;
    use vortex::array::arrays::BoolArray;
    use vortex::array::arrays::MaskedArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::validity::Validity;
    use vortex::error::VortexExpect;
    use vortex::error::VortexResult;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    #[crate::test]
    async fn test_cuda_masked_primitive() -> VortexResult<()> {
        let mut ctx = vortex::array::array_session().create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())
            .vortex_expect("failed to create execution context");
        let validity =
            BoolArray::from_iter([false, false, true, false, true, true, false, false, false])
                .into_array()
                .slice(2..7)?;
        let masked = MaskedArray::try_new(
            PrimitiveArray::from_iter([10i64, 20, 30, 40, 50]).into_array(),
            Validity::Array(validity),
        )?;

        let actual = MaskedExecutor
            .execute(masked.clone().into_array(), &mut cuda_ctx)
            .await?
            .into_host()
            .await?
            .into_array();

        assert_arrays_eq!(masked, actual, &mut ctx);
        Ok(())
    }
}
