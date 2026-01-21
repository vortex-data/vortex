// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::arrays::DictVTable;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;

use crate::executor::CudaExecute;
use crate::executor::CudaExecutionCtx;

/// CUDA executor for dictionary-encoded arrays.
#[derive(Debug)]
pub struct DictExecutor;

#[async_trait]
impl CudaExecute for DictExecutor {
    async fn execute(
        &self,
        array: ArrayRef,
        _ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let _dict_array = array
            .try_into::<DictVTable>()
            .ok()
            .vortex_expect("Array is not a Dict array");

        todo!()
    }
}
