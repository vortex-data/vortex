// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;

use num_traits::ToPrimitive;
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
    let lengths = fsst
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let validity = fsst.validity()?.execute_mask(fsst.len(), ctx)?;

    // Only the valid rows too long to inline are referenced by the views.
    let data_bytes = referenced_len(&lengths, &validity)?;

    fsst.len()
        .checked_mul(size_of::<BinaryView>())
        .and_then(|size| size.checked_add(data_bytes))
        .and_then(|size| u64::try_from(size).ok())
        .and_then(|size| size.checked_add(validity_size(validity)))
        .ok_or_else(|| vortex_err!("FSST uncompressed size overflowed"))
}

/// Sums the lengths of the valid rows that a [`BinaryView`] cannot inline.
fn referenced_len(lengths: &PrimitiveArray, validity: &Mask) -> VortexResult<usize> {
    match_each_integer_ptype!(lengths.ptype(), |P| {
        let lengths = lengths.as_slice::<P>();
        let out_of_line = |acc: usize, &length: &P| {
            let length = length.to_usize()?;
            if length > BinaryView::MAX_INLINED_SIZE {
                acc.checked_add(length)
            } else {
                Some(acc)
            }
        };
        match validity {
            Mask::AllTrue(_) => lengths.iter().try_fold(0usize, out_of_line),
            Mask::AllFalse(_) => Some(0),
            Mask::Values(values) => values
                .indices()
                .iter()
                .try_fold(0usize, |acc, &row| out_of_line(acc, &lengths[row])),
        }
    })
    .ok_or_else(|| vortex_err!("FSST uncompressed lengths are negative or overflow"))
}

fn validity_size(validity: Mask) -> u64 {
    match validity {
        Mask::AllTrue(_) => 0,
        Mask::AllFalse(length) => ConstantArray::new(false, length).into_array().nbytes(),
        Mask::Values(values) => values.len().div_ceil(u8::BITS as usize) as u64,
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rstest::rstest;
    use vortex_array::ArrayRef;
    use vortex_array::ArrayVTable;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::aggregate_fn::AggregateFnVTable;
    use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
    use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
    use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::FSST;
    use crate::canonical::total_uncompressed_len;
    use crate::fsst_compress;
    use crate::fsst_train_compressor;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    /// Compresses `values` to FSST, then slices it to `range` when one is given.
    fn encode(
        values: Vec<Option<&str>>,
        range: Option<std::ops::Range<usize>>,
    ) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = VarBinArray::from_iter(values, DType::Utf8(Nullability::Nullable)).into_array();
        let compressor = fsst_train_compressor(&input, &mut ctx)?;
        let encoded = fsst_compress(&input, &compressor, &mut ctx)?.into_array();
        match range {
            Some(range) => encoded.slice(range),
            None => Ok(encoded),
        }
    }

    const OUTLINED: &str = "a string long enough to use an outlined view";

    #[rstest]
    #[case::inlined_and_outlined(vec![Some("short"), Some(OUTLINED), Some("")], None)]
    #[case::nullable(vec![Some("alpha"), None, Some(OUTLINED), None], None)]
    #[case::all_invalid(vec![None, None, None], None)]
    #[case::sliced(
        vec![Some("zero"), None, Some(OUTLINED), Some("three"), None, Some(OUTLINED)],
        Some(1..5),
    )]
    fn matches_decoded_size(
        #[case] values: Vec<Option<&str>>,
        #[case] range: Option<std::ops::Range<usize>>,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let encoded = encode(values, range)?;
        let decoded = encoded
            .clone()
            .execute::<VarBinViewArray>(&mut ctx)?
            .into_array();

        assert_eq!(
            uncompressed_size_in_bytes(&encoded, &mut ctx)?,
            uncompressed_size_in_bytes(&decoded, &mut ctx)?,
        );
        Ok(())
    }

    #[test]
    fn registers_kernel() {
        assert!(
            SESSION
                .aggregate_fns()
                .find_aggregate_kernel(FSST.id(), UncompressedSizeInBytes.id())
                .is_some()
        );
    }

    #[rstest]
    #[case::negative(PrimitiveArray::from_iter([-1i64]))]
    #[case::overflow(PrimitiveArray::from_iter([u64::MAX, 1]))]
    fn rejects_invalid_lengths(#[case] lengths: PrimitiveArray) {
        assert!(total_uncompressed_len(&lengths).is_err());
    }
}
