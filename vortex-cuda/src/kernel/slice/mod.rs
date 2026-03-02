// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::DynArray;
use vortex::array::arrays::SliceArrayParts;
use vortex::array::arrays::SliceVTable;
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
        let slice_array = array.try_into::<SliceVTable>().map_err(|array| {
            vortex_err!(
                "SliceExecutor requires input of SliceArray, was {}",
                array.encoding_id()
            )
        })?;

        let SliceArrayParts { child, range } = slice_array.into_parts();
        let child = child.execute_cuda(ctx).await?;

        match child {
            Canonical::Null(null_array) => null_array.slice(range)?.to_canonical(),
            Canonical::Bool(bool_array) => bool_array.slice(range)?.to_canonical(),
            Canonical::Primitive(prim_array) => prim_array.slice(range)?.to_canonical(),
            Canonical::Decimal(decimal_array) => decimal_array.slice(range)?.to_canonical(),
            Canonical::VarBinView(varbinview) => varbinview.slice(range)?.to_canonical(),
            Canonical::Extension(extension_array) => extension_array.slice(range)?.to_canonical(),
            c => todo!("Slice kernel not implemented for {}", c.dtype()),
        }
    }
}
