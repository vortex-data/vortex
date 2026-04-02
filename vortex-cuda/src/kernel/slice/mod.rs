// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use async_trait::async_trait;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::IntoArray;
use vortex::array::arrays::Slice;
use vortex::array::arrays::slice::SliceArrayParts;
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
        let slice_array = array.try_into::<Slice>().map_err(|array| {
            vortex_err!(
                "SliceExecutor requires input of SliceArray, was {}",
                array.encoding_id()
            )
        })?;

        let SliceArrayParts { child, range } = slice_array.into_data().into_parts();
        let child = child.execute_cuda(ctx).await?;

        match child {
            Canonical::Null(null_array) => null_array.into_array().slice(range)?.to_canonical(),
            Canonical::Bool(bool_array) => bool_array.into_array().slice(range)?.to_canonical(),
            Canonical::Primitive(prim_array) => {
                prim_array.into_array().slice(range)?.to_canonical()
            }
            Canonical::Decimal(decimal_array) => {
                decimal_array.into_array().slice(range)?.to_canonical()
            }
            Canonical::VarBinView(varbinview) => {
                varbinview.into_array().slice(range)?.to_canonical()
            }
            Canonical::Extension(extension_array) => {
                extension_array.into_array().slice(range)?.to_canonical()
            }
            c => todo!("Slice kernel not implemented for {}", c.dtype()),
        }
    }
}
