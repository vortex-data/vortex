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
use vortex::encodings::decimal_byte_parts::DecimalBytePartsArraySlotsExt;
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

        // Reassembling lower parts into wide decimals is not implemented on the GPU; the MSP
        // alone is not the value.
        if !array.lower_parts().is_empty() {
            vortex_bail!("DecimalBytePartsArray with lower parts is not supported on GPU")
        }

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
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::arrays::WidenArray;
    use vortex::array::assert_arrays_eq;
    use vortex::dtype::DType;
    use vortex::dtype::DecimalDType;
    use vortex::dtype::NativePType;
    use vortex::dtype::PType;
    use vortex::encodings::decimal_byte_parts::DecimalByteParts;
    use vortex::error::VortexResult;
    use vortex::session::VortexSession;
    use vortex_array::VortexSessionExecute;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    /// The stored MSP values of every case: they fit precision 2, and one row is null.
    fn fixture<T: NativePType + From<i8>>() -> PrimitiveArray {
        PrimitiveArray::from_option_iter(
            [Some(-99i8), None, Some(42), Some(99)].map(|value| value.map(<T as From<i8>>::from)),
        )
    }

    /// Each case stores the MSP at `stored`, built by [`fixture`], and gives the logical MSP ptype
    /// of the precision. A narrower stored MSP is wrapped in a `Widen`.
    #[rstest]
    #[case::i8_p2(fixture::<i8>(),    2,  PType::I8)]
    #[case::i8_p5(fixture::<i8>(),    5,  PType::I32)]
    #[case::i64_p38(fixture::<i64>(), 38, PType::I64)]
    #[case::i16_p76(fixture::<i16>(), 76, PType::I64)]
    #[crate::test]
    async fn test_decimal_byte_parts_gpu_decode(
        #[case] stored: PrimitiveArray,
        #[case] precision: u8,
        #[case] logical_ptype: PType,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())?;

        let decimal_dtype = DecimalDType::new(precision, 2);
        let nullability = stored.dtype().nullability();
        let msp = if stored.ptype() == logical_ptype {
            stored.into_array()
        } else {
            WidenArray::try_new(
                stored.into_array(),
                DType::Primitive(logical_ptype, nullability),
            )?
            .into_array()
        };
        let dbp_array = DecimalByteParts::try_new(msp, decimal_dtype)?.into_array();

        let gpu_result = DecimalBytePartsExecutor
            .execute(dbp_array.clone(), &mut cuda_ctx)
            .await?
            .into_host(cuda_ctx.execution_ctx())
            .await?
            .into_decimal();

        assert_arrays_eq!(gpu_result, dbp_array, &mut ctx);

        Ok(())
    }
}
