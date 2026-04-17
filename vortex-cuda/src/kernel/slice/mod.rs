// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::arrays::Slice;
use vortex::array::arrays::slice::SliceArrayExt;
use vortex::error::VortexResult;
use vortex::error::vortex_err;

use crate::CudaExecutionCtx;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;

#[derive(Debug)]
pub struct SliceExecutor;

#[async_trait]
impl CudaExecute for SliceExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let slice_array = array.try_downcast::<Slice>().map_err(|array| {
            vortex_err!(
                "SliceExecutor requires input of SliceArray, was {}",
                array.encoding_id()
            )
        })?;

        let range = slice_array.data().slice_range().clone();
        let child = slice_array.child().clone().execute_cuda(ctx).await?;

        match child {
            Canonical::Null(null_array) => null_array
                .into_array()
                .slice(range)?
                .execute::<Canonical>(ctx.execution_ctx()),
            Canonical::Bool(bool_array) => bool_array
                .into_array()
                .slice(range)?
                .execute::<Canonical>(ctx.execution_ctx()),
            Canonical::Primitive(prim_array) => prim_array
                .into_array()
                .slice(range)?
                .execute::<Canonical>(ctx.execution_ctx()),
            Canonical::Decimal(decimal_array) => decimal_array
                .into_array()
                .slice(range)?
                .execute::<Canonical>(ctx.execution_ctx()),
            Canonical::VarBinView(varbinview) => varbinview
                .into_array()
                .slice(range)?
                .execute::<Canonical>(ctx.execution_ctx()),
            Canonical::Extension(extension_array) => extension_array
                .into_array()
                .slice(range)?
                .execute::<Canonical>(ctx.execution_ctx()),
            c => todo!("Slice kernel not implemented for {}", c.dtype()),
        }
    }
}
