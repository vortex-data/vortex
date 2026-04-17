// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::Shared;
use vortex::array::arrays::shared::SharedArrayExt;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;

use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;

/// CUDA executor for SharedArray.
#[derive(Debug)]
pub(crate) struct SharedExecutor;

#[async_trait]
impl CudaExecute for SharedExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let shared = array
            .try_downcast::<Shared>()
            .ok()
            .vortex_expect("Array is not a Shared array");

        shared
            .get_or_compute_async(|source| source.execute_cuda(ctx))
            .await?
            .execute::<Canonical>(ctx.execution_ctx())
    }
}
