// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

use std::mem::size_of;

use num_traits::AsPrimitive;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::AggregateFnRef;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeInBytes;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeOpts;
use vortex_array::aggregate_fn::kernels::DynAggregateKernel;
use vortex_array::arrays::ConstantArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::varbinview::BinaryView;
use vortex_array::match_each_integer_ptype;
use vortex_array::scalar::Scalar;
use vortex_error::VortexResult;
use vortex_error::vortex_err;
use vortex_mask::Mask;

use crate::OnPair;
use crate::OnPairArraySlotsExt;
use crate::canonical::total_uncompressed_len;

/// Computes the canonical OnPair size from stored lengths without decoding the string payload.
///
/// The stored lengths are authoritative because verifying them requires decoding the payload.
#[derive(Debug)]
pub(crate) struct OnPairUncompressedSizeKernel;

impl DynAggregateKernel for OnPairUncompressedSizeKernel {
    fn aggregate(
        &self,
        aggregate_fn: &AggregateFnRef,
        batch: &ArrayRef,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<Option<Scalar>> {
        let Some(opts) = aggregate_fn.as_opt::<UncompressedSizeInBytes>() else {
            return Ok(None);
        };

        let Some(onpair) = batch.as_opt::<OnPair>() else {
            return Ok(None);
        };

        Ok(Some(Scalar::from(uncompressed_size(onpair, *opts, ctx)?)))
    }
}

fn uncompressed_size(
    onpair: ArrayView<'_, OnPair>,
    opts: UncompressedSizeOpts,
    ctx: &mut ExecutionCtx,
) -> VortexResult<u64> {
    let lengths = onpair
        .uncompressed_lengths()
        .clone()
        .execute::<PrimitiveArray>(ctx)?;
    let validity = onpair.validity()?.execute_mask(onpair.len(), ctx)?;

    // Canonicalization decodes every row into data buffers, so they hold the sum of the lengths.
    // Only the valid rows too long to inline are referenced by the views.
    let data_bytes = if opts.is_exact() {
        referenced_len(&lengths, &validity)?
    } else {
        total_uncompressed_len(&lengths)?
    };

    onpair.len()
        .checked_mul(size_of::<BinaryView>())
        .and_then(|size| size.checked_add(data_bytes))
        .and_then(|size| u64::try_from(size).ok())
        .and_then(|size| size.checked_add(validity_size(validity)))
        .ok_or_else(|| vortex_err!("OnPair uncompressed size overflowed"))
}

/// Sums the lengths of the valid rows that a [`BinaryView`] cannot inline.
fn referenced_len(lengths: &PrimitiveArray, validity: &Mask) -> VortexResult<usize> {
    match_each_integer_ptype!(lengths.ptype(), |P| {
        let lengths = lengths.as_slice::<P>();
        let out_of_line = |acc: usize, &length: &P| {
            let length: usize = length.as_();
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
    .ok_or_else(|| vortex_err!("OnPair uncompressed lengths overflow"))
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
    use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::UncompressedSizeOpts;
    use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
    use vortex_array::aggregate_fn::session::AggregateFnSessionExt;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::VarBinArray;
    use vortex_array::arrays::VarBinViewArray;
    use vortex_array::dtype::DType;
    use vortex_array::dtype::Nullability;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;

    use crate::DEFAULT_CONFIG;
    use crate::OnPair;
    use crate::canonical::total_uncompressed_len;
    use crate::onpair_compress;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty();
        crate::initialize(&session);
        session
    });

    /// Compresses `values` to OnPair, then slices it to `range` when one is given.
    fn encode(
        values: Vec<Option<&str>>,
        range: Option<std::ops::Range<usize>>,
    ) -> VortexResult<ArrayRef> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = VarBinArray::from_iter(values, DType::Utf8(Nullability::Nullable)).into_array();
        let encoded = onpair_compress(&input, DEFAULT_CONFIG, &mut ctx)?;
        assert!(encoded.is::<OnPair>());
        match range {
            Some(range) => encoded.slice(range),
            None => Ok(encoded),
        }
    }

    const OUTLINED: &str = "a string long enough to use an outlined view";

    #[rstest]
    #[case::inlined_and_outlined(vec![Some("short"), Some(OUTLINED), Some("")], None)]
    #[case::nullable(vec![Some("alpha"), None, Some(OUTLINED), None], None)]
    // OnPair compresses an all-null input to a constant, so slice one out of a mixed array.
    #[case::all_invalid(vec![Some(OUTLINED), None, None, Some("x")], Some(1..3))]
    #[case::sliced(
        vec![Some("zero"), None, Some(OUTLINED), Some("three"), None, Some(OUTLINED)],
        Some(1..5),
    )]
    fn matches_decoded_size(
        #[case] values: Vec<Option<&str>>,
        #[case] range: Option<std::ops::Range<usize>>,
        #[values(UncompressedSizeOpts::inexact(), UncompressedSizeOpts::exact())]
        opts: UncompressedSizeOpts,
    ) -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let encoded = encode(values, range)?;
        let decoded = encoded
            .clone()
            .execute::<VarBinViewArray>(&mut ctx)?
            .into_array();

        assert_eq!(
            uncompressed_size_in_bytes(&encoded, opts, &mut ctx)?,
            uncompressed_size_in_bytes(&decoded, opts, &mut ctx)?,
        );
        Ok(())
    }

    #[test]
    fn registers_kernel() {
        assert!(
            SESSION
                .aggregate_fns()
                .find_aggregate_kernel(OnPair.id(), UncompressedSizeInBytes.id())
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
