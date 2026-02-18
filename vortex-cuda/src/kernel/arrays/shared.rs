// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use tracing::instrument;
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
pub(crate) struct SharedExecutor;

impl CudaExecute for SharedExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    fn execute(&self, array: ArrayRef, ctx: &mut CudaExecutionCtx) -> VortexResult<Canonical> {
        let shared = array
            .try_into::<SharedVTable>()
            .ok()
            .vortex_expect("Array is not a Shared array");

        shared.get_or_compute(|source| source.clone().execute_cuda(ctx))
    }
}
