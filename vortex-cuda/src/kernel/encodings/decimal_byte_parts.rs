// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use async_trait::async_trait;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::arrays::DecimalArray;
use vortex_array::arrays::PrimitiveArrayParts;
use vortex_decimal_byte_parts::DecimalBytePartsArrayParts;
use vortex_decimal_byte_parts::DecimalBytePartsVTable;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;

use crate::CudaExecutionCtx;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;

// See `DecimalBytePartsArray`
#[derive(Debug)]
pub struct DecimalBytePartsExecutor;

#[async_trait]
impl CudaExecute for DecimalBytePartsExecutor {
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let Ok(array) = array.try_into::<DecimalBytePartsVTable>() else {
            vortex_bail!("cannot downcast to DecimalBytePartsArray")
        };

        let decimal_dtype = array.decimal_dtype().clone();
        let DecimalBytePartsArrayParts { msp, .. } = array.into_parts();
        let PrimitiveArrayParts {
            buffer,
            ptype,
            validity,
            ..
        } = msp.execute_cuda(ctx).await?.into_primitive().into_parts();

        // SAFETY: The primitive array's buffer is already validated with correct type.
        // The decimal dtype matches the array's dtype, and validity is preserved.
        Ok(Canonical::Decimal(unsafe {
            DecimalArray::new_unchecked_handle(buffer, ptype.try_into()?, decimal_dtype, validity)
        }))
    }
}
