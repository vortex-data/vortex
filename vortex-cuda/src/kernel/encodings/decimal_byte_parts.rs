// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use async_trait::async_trait;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::encodings::decimal_byte_parts::DecimalByteParts;
use vortex::encodings::decimal_byte_parts::DecimalBytePartsArrayExt;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;

use crate::CudaExecutionCtx;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;

// See `DecimalBytePartsArray`
#[derive(Debug)]
pub(crate) struct DecimalBytePartsExecutor;

#[async_trait]
impl CudaExecute for DecimalBytePartsExecutor {
    #[instrument(level = "trace", skip_all, fields(executor = ?self))]
    async fn execute(
        &self,
        array: ArrayRef,
        ctx: &mut CudaExecutionCtx,
    ) -> VortexResult<Canonical> {
        let Ok(array) = array.try_downcast::<DecimalByteParts>() else {
            vortex_bail!("cannot downcast to DecimalBytePartsArray")
        };

        let decimal_dtype = *array
            .dtype()
            .as_decimal_opt()
            .vortex_expect("DecimalBytePartsArray dtype must be decimal");
        let msp = array.msp().clone();
        let PrimitiveDataParts {
            buffer,
            ptype,
            validity,
            ..
        } = msp
            .execute_cuda(ctx)
            .await?
            .into_primitive()
            .into_data_parts();

        // SAFETY: The primitive array's buffer is already validated with correct type.
        // The decimal dtype matches the array's dtype, and validity is preserved.
        Ok(Canonical::Decimal(unsafe {
            DecimalArray::new_unchecked_handle(buffer, ptype.try_into()?, decimal_dtype, validity)
        }))
    }
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::dtype::DecimalDType;
    use vortex::encodings::decimal_byte_parts::DecimalByteParts;
    use vortex::error::VortexExpect;
    use vortex::session::VortexSession;

    use super::*;
    use crate::session::CudaSession;

    #[rstest]
    #[case::i8_p5_s2(Buffer::from(vec![100i8, 101, 102, -1, -100]), 5, 2)]
    #[case::i16_p10_s2(Buffer::from(vec![100i16, 200, 300, 400, 500]), 10, 2)]
    #[case::i32_p18_s4(Buffer::from(vec![100i32, 200, 300, 400, 500]), 18, 4)]
    #[case::i64_p38_s6(Buffer::from(vec![100i64, 200, 300, 400, 500]), 38, 6)]
    #[crate::test]
    async fn test_decimal_byte_parts_gpu_decode<T: vortex::dtype::NativePType>(
        #[case] encoded: Buffer<T>,
        #[case] precision: u8,
        #[case] scale: i8,
    ) {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&VortexSession::empty())
            .vortex_expect("create execution context");

        let decimal_dtype = DecimalDType::new(precision, scale);
        let dbp_array = DecimalByteParts::try_new(
            PrimitiveArray::new(encoded, Validity::NonNullable).into_array(),
            decimal_dtype,
        )
        .vortex_expect("create DecimalBytePartsArray");

        let cpu_result =
            crate::canonicalize_cpu(dbp_array.clone()).vortex_expect("CPU canonicalize");

        let gpu_result = DecimalBytePartsExecutor
            .execute(dbp_array.into_array(), &mut cuda_ctx)
            .await
            .vortex_expect("GPU decode");

        assert_arrays_eq!(cpu_result.into_array(), gpu_result.into_array());
    }
}
