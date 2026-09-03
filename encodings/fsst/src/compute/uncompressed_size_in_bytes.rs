// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::scalar::Scalar;
use vortex_array::validity::Validity;
use vortex_error::VortexResult;
use vortex_error::vortex_err;

use crate::FSST;
use crate::canonical::FsstDecodePlan;

#[derive(Debug)]
pub(crate) struct FsstUncompressedSizeKernel;

impl DynAggregateKernel for FsstUncompressedSizeKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        if !aggregate_fn.is::<UncompressedSizeInBytes>() {
            return Ok(None);
        }
        let Some(fsst) = batch.as_opt::<FSST>() else {
            return Ok(None);
        };
        Ok(Some(Scalar::from(uncompressed_size(fsst, ctx)?)))
    }
}

fn uncompressed_size(fsst: ArrayView<'_, FSST>, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    let plan = FsstDecodePlan::new(fsst, ctx)?;
    let views_size = fsst
        .len()
        .checked_mul(size_of::<BinaryView>())
        .ok_or_else(|| vortex_err!("FSST view size overflow"))?;
    let validity_size = match fsst.validity()? {
        Validity::NonNullable | Validity::AllValid | Validity::AllInvalid => 0,
        Validity::Array(validity) => validity.len().div_ceil(u8::BITS as usize),
    };
    views_size
        .checked_add(plan.total_size)
        .and_then(|size| size.checked_add(validity_size))
        .and_then(|size| u64::try_from(size).ok())
        .ok_or_else(|| vortex_err!("FSST uncompressed size overflow"))
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
    use vortex_array::array_session;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_error::VortexResult;

    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    #[test]
    fn matches_canonical_size_for_nullable_strings() -> VortexResult<()> {
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let input = VarBinArray::from_iter(
            [
                Some("short"),
                None,
                Some("a string that uses an outlined view"),
                Some("another outlined string"),
            ],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        let compressor = fsst_train_compressor(&input, &mut ctx)?;
        let encoded = fsst_compress(&input, &compressor, &mut ctx)?.into_array();
        let expected = uncompressed_size_in_bytes(&input, &mut ctx)?;
        let actual = uncompressed_size_in_bytes(&encoded, &mut ctx)?;
        assert_eq!(actual, expected);
        Ok(())
    }
}
