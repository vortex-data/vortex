// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::fmt::Debug;

use async_trait::async_trait;
use tracing::instrument;
use vortex::array::ArrayRef;
use vortex::array::Canonical;
use vortex::array::arrays::DecimalArray;
use vortex::array::arrays::primitive::PrimitiveDataParts;
use vortex::dtype::DecimalType;
use vortex::encodings::decimal_byte_parts::DecimalByteParts;
use vortex::encodings::decimal_byte_parts::DecimalBytePartsArraySlotsExt;
use vortex::error::VortexExpect;
use vortex::error::VortexResult;
use vortex::error::vortex_bail;
use vortex::error::vortex_ensure;

use crate::CudaExecutionCtx;
use crate::executor::CudaArrayExt;
use crate::executor::CudaExecute;
use crate::kernel::widen_decimal_values;

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
        let len = msp.len();
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

        // NB: Narrowing the MSP must check every valid value against the precision, and that check
        // on the device needs an overflow flag that synchronizes with the host. Older writers can
        // store an MSP wider than the canonical width, because they narrowed a decimal by the min
        // and max of every slot, null slots included. The CUDA Arrow export already rejects
        // storage wider than the Arrow width, so such a column failed on the GPU before this
        // check as well.
        let msp_type = DecimalType::try_from(ptype)?;
        let canonical_type = decimal_dtype.canonical_values_type();
        vortex_ensure!(
            msp_type <= canonical_type,
            "DecimalBytePartsArray on GPU requires an MSP no wider than the canonical width \
             {canonical_type}, got {msp_type}"
        );

        // An MSP at the canonical width already holds the decimal values, so it needs no copy.
        // The CPU decoder returns a `Widen` over a narrow MSP, but CUDA has no `Widen` kernel, so
        // the device widens to the canonical width.
        let values = if msp_type == canonical_type {
            buffer
        } else {
            widen_decimal_values(buffer, msp_type, canonical_type, len, ctx).await?
        };

        // SAFETY: The values hold one value per row of the MSP at the canonical width of
        // `decimal_dtype`, so the validity of the MSP fits them. A widening copy keeps every
        // value, and reads trust the values of the MSP to be within the precision.
        Ok(Canonical::Decimal(unsafe {
            DecimalArray::new_unchecked_handle(values, canonical_type, decimal_dtype, validity)
        }))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex::array::IntoArray;
    use vortex::array::arrays::PrimitiveArray;
    use vortex::array::assert_arrays_eq;
    use vortex::array::validity::Validity;
    use vortex::buffer::Buffer;
    use vortex::dtype::DecimalDType;
    use vortex::encodings::decimal_byte_parts::DecimalByteParts;
    use vortex::error::VortexResult;
    use vortex::session::VortexSession;
    use vortex_array::VortexSessionExecute;

    use super::*;
    use crate::CanonicalCudaExt;
    use crate::session::CudaSession;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(VortexSession::empty);

    #[rstest]
    #[case::i8_p3_s0(Buffer::from(vec![99i8, 1, 0, -1, -99]), 3, 0)]
    #[case::i8_p5_s2(Buffer::from(vec![100i8, 101, 102, -1, -100]), 5, 2)]
    #[case::i16_p10_s2(Buffer::from(vec![100i16, 200, 300, 400, 500]), 10, 2)]
    #[case::i32_p9_s2(Buffer::from(vec![100i32, 200, 300, -400, -500]), 9, 2)]
    #[case::i32_p18_s4(Buffer::from(vec![100i32, 200, 300, 400, 500]), 18, 4)]
    #[case::i64_p38_s6(Buffer::from(vec![100i64, 200, 300, 400, 500]), 38, 6)]
    #[case::i64_p40_s2(Buffer::from(vec![i64::MAX, 1, 0, -1, i64::MIN]), 40, 2)]
    #[crate::test]
    async fn test_decimal_byte_parts_gpu_decode<T: vortex::dtype::NativePType>(
        #[case] encoded: Buffer<T>,
        #[case] precision: u8,
        #[case] scale: i8,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())?;

        let decimal_dtype = DecimalDType::new(precision, scale);
        let dbp_array = DecimalByteParts::try_new(
            PrimitiveArray::new(encoded, Validity::NonNullable).into_array(),
            decimal_dtype,
        )?;

        let gpu_result = DecimalBytePartsExecutor
            .execute(dbp_array.clone().into_array(), &mut cuda_ctx)
            .await?
            .into_host(cuda_ctx.execution_ctx())
            .await?
            .into_decimal();

        assert_eq!(
            gpu_result.values_type(),
            decimal_dtype.canonical_values_type()
        );
        assert_arrays_eq!(gpu_result, dbp_array, &mut ctx);
        Ok(())
    }

    #[crate::test]
    async fn test_decimal_byte_parts_gpu_decode_rejects_msp_wider_than_canonical()
    -> VortexResult<()> {
        let mut cuda_ctx = CudaSession::create_execution_ctx(&crate::cuda_session())?;

        let dbp_array = DecimalByteParts::try_new(
            PrimitiveArray::new(Buffer::from(vec![99i64, -99]), Validity::NonNullable).into_array(),
            DecimalDType::new(4, 0),
        )?;

        let err = DecimalBytePartsExecutor
            .execute(dbp_array.into_array(), &mut cuda_ctx)
            .await
            .unwrap_err();
        assert!(
            err.to_string()
                .contains("requires an MSP no wider than the canonical width")
        );
        Ok(())
    }
}
