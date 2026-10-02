// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::FSST;
use crate::FSSTArraySlotsExt;

/// Computes the canonical FSST size from stored lengths without decoding the string payload.
///
/// The stored lengths are authoritative because verifying them requires decoding the payload.
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
    let decoded_bytes = decoded_size(fsst, ctx)?;
    let validity_size = validity_size(fsst.validity()?.execute_mask(fsst.len(), ctx)?)?;

    canonical_size(fsst.len(), decoded_bytes, validity_size)
}

fn decoded_size(fsst: ArrayView<'_, FSST>, ctx: &mut ExecutionCtx) -> VortexResult<u64> {
    let lengths = fsst
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;

    match_each_integer_ptype!(lengths.ptype(), |P| {
        checked_sum_lengths(lengths.as_slice::<P>())
    })
}

fn checked_sum_lengths<P>(lengths: &[P]) -> VortexResult<u64>
where
    P: Copy,
    u64: TryFrom<P>,
    <u64 as TryFrom<P>>::Error: std::fmt::Display,
{
    lengths.iter().try_fold(0u64, |total, length| {
        let length = u64::try_from(*length).map_err(|error| {
            vortex_err!("Failed to convert FSST uncompressed length to u64: {error}")
        })?;
        total
            .checked_add(length)
            .ok_or_else(|| vortex_err!("FSST decoded size overflowed u64"))
    })
}

fn validity_size(validity: Mask) -> VortexResult<u64> {
    match validity {
        Mask::AllTrue(_) => Ok(0),
        Mask::AllFalse(length) => Ok(ConstantArray::new(false, length).into_array().nbytes()),
        Mask::Values(values) => u64::try_from(values.len().div_ceil(u8::BITS as usize))
            .map_err(|error| vortex_err!("Failed to convert FSST validity size to u64: {error}")),
    }
}

fn canonical_size(length: usize, decoded_bytes: u64, validity_bytes: u64) -> VortexResult<u64> {
    let views_size =
        u64::try_from(length)
            .map_err(|error| vortex_err!("Failed to convert FSST length to u64: {error}"))?
            .checked_mul(u64::try_from(size_of::<BinaryView>()).map_err(|error| {
                vortex_err!("Failed to convert binary view width to u64: {error}")
            })?)
            .ok_or_else(|| vortex_err!("FSST uncompressed size overflowed u64"))?;
    views_size
        .checked_add(decoded_bytes)
        .and_then(|size| size.checked_add(validity_bytes))
        .ok_or_else(|| vortex_err!("FSST uncompressed size overflowed u64"))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::ArrayVTable;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
    use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
    use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
    use vortex_array::array_session;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_error::VortexResult;

    use super::checked_sum_lengths;
    use crate::FSST;
    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    fn assert_matches_materialized_size(array: ArrayRef) -> VortexResult<()> {
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let compressor = fsst_train_compressor(&array, &mut ctx)?;
        let encoded = fsst_compress(&array, &compressor, &mut ctx)?.into_array();

        assert_encoded_size_matches_materialized(&encoded, &mut ctx)
    }

    fn assert_encoded_size_matches_materialized(
        encoded: &ArrayRef,
        ctx: &mut vortex_array::ExecutionCtx,
    ) -> VortexResult<()> {
        let actual = u64::try_from(uncompressed_size_in_bytes(encoded, ctx)?)?;
        let materialized = encoded.clone().execute::<VarBinViewArray>(ctx)?;

        assert_eq!(actual, materialized.as_ref().nbytes());
        Ok(())
    }

    #[rstest]
    #[case::utf8(
        vec![
            Some(b"short".to_vec()),
            Some(b"a string that uses an outlined view".to_vec()),
            Some(Vec::new()),
        ],
        DType::Utf8(Nullability::NonNullable),
    )]
    #[case::nullable_utf8(
        vec![
            Some(b"alpha".to_vec()),
            None,
            Some(b"a nullable outlined string".to_vec()),
        ],
        DType::Utf8(Nullability::Nullable),
    )]
    #[case::binary(
        vec![
            Some(vec![0, 159, 146, 150]),
            Some(vec![255; 24]),
            Some(Vec::new()),
        ],
        DType::Binary(Nullability::NonNullable),
    )]
    #[case::all_invalid(
        vec![None, None, None],
        DType::Utf8(Nullability::Nullable),
    )]
    fn matches_materialized_size(
        #[case] values: Vec<Option<Vec<u8>>>,
        #[case] dtype: DType,
    ) -> VortexResult<()> {
        assert_matches_materialized_size(VarBinArray::from_iter(values, dtype).into_array())
    }

    #[test]
    fn registers_kernel() {
        let session = array_session();
        crate::initialize(&session);

        assert!(
            session
                .aggregate_fns()
                .find_aggregate_kernel(FSST.id(), UncompressedSizeInBytes.id())
                .is_some()
        );
    }

    #[test]
    fn sliced_nullable_array_matches_materialized_size() -> VortexResult<()> {
        let session = array_session();
        crate::initialize(&session);
        let mut ctx = session.create_execution_ctx();
        let input = VarBinArray::from_iter(
            [
                Some("zero"),
                None,
                Some("two uses an outlined string value"),
                Some("three"),
                None,
            ],
            DType::Utf8(Nullability::Nullable),
        )
        .into_array();
        let compressor = fsst_train_compressor(&input, &mut ctx)?;
        let encoded = fsst_compress(&input, &compressor, &mut ctx)?
            .into_array()
            .slice(1..5)?;

        assert_encoded_size_matches_materialized(&encoded, &mut ctx)
    }

    #[rstest]
    #[case::negative(checked_sum_lengths(&[-1i64]))]
    #[case::overflow(checked_sum_lengths(&[u64::MAX, 1]))]
    fn rejects_invalid_lengths(#[case] result: VortexResult<u64>) {
        assert!(result.is_err());
    }
}
