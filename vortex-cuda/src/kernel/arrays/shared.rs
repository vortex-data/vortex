// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::arrays::SharedVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;

/// CUDA executor for SharedArray.
#[derive(Debug)]
pub struct SharedExecutor;

#[async_trait]
impl CudaExecute for SharedExecutor {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let shared = array
            .try_into::<SharedVTable>()
            .ok()
            .vortex_expect("Array is not a Shared array");

        if let Some(cached) = shared.cached() {
            return Ok(cached);
        }

        let source = shared
            .source_if_any()
            .vortex_expect("not cached shared array has to have the source array");
        let canonical = source.execute_cuda(ctx).await?;
        Ok(shared.cache_or_return(canonical))
    }
}
