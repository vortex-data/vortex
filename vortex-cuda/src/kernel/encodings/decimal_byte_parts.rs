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

        let decimal_dtype = *array.decimal_dtype();
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

#[cfg(test)]
#[cfg(cuda_available)]
mod tests {
    use rstest::rstest;
    use vortex_array::IntoArray;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_decimal_byte_parts::DecimalBytePartsArray;
    use vortex_dtype::DecimalDType;
    use vortex_error::VortexExpect;
    use vortex_session::VortexSession;

    use super::*;
    use crate::session::CudaSession;

    #[rstest]
    #[case::i8_p5_s2(Buffer::from(vec![100i8, 101, 102, -1, -100]), 5, 2)]
    #[case::i16_p10_s2(Buffer::from(vec![100i16, 200, 300, 400, 500]), 10, 2)]
    #[case::i32_p18_s4(Buffer::from(vec![100i32, 200, 300, 400, 500]), 18, 4)]
    #[case::i64_p38_s6(Buffer::from(vec![100i64, 200, 300, 400, 500]), 38, 6)]
    #[tokio::test]
    async fn test_decimal_byte_parts_gpu_decode<T: vortex_dtype::NativePType>(
        #[case] encoded: Buffer<T>,
        #[case] precision: u8,
        #[case] scale: i8,
    ) {
        let mut cuda_ctx = CudaSession::create_execution_ctx(VortexSession::empty())
            .vortex_expect("create execution context");

        let decimal_dtype = DecimalDType::new(precision, scale);
        let dbp_array = DecimalBytePartsArray::try_new(
            PrimitiveArray::new(encoded, Validity::NonNullable).into_array(),
            decimal_dtype,
        )
        .vortex_expect("create DecimalBytePartsArray");

        let cpu_result = dbp_array.to_canonical().vortex_expect("CPU canonicalize");

        let gpu_result = DecimalBytePartsExecutor
            .execute(dbp_array.to_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decode");

        assert_arrays_eq!(cpu_result.into_array(), gpu_result.into_array());
    }
}
