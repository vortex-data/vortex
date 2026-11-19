// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Elias-Fano encoding for sorted unsigned integers.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::aggregate_fn::fns::is_sorted::is_sorted;
use vortex_array::arrays::PrimitiveArray;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateScore;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_elias_fano::EliasFano;
use vortex_elias_fano::EliasFanoArrayExt;
use vortex_elias_fano::EliasFanoArraySlotsExt;
use vortex_elias_fano::Unary;
use vortex_elias_fano::elias_fano_encode;
use vortex_elias_fano::unary_encode;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_error::vortex_err;
use vortex_fastlanes::FL_CHUNK_SIZE;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArrayExt;
use vortex_fastlanes::FoRArraySlotsExt;
use vortex_fastlanes::FoRReferences;
use vortex_fastlanes::for_v1_id;

use super::BITPACKING_V1;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;

/// Elias-Fano encoding for sorted, non-nullable unsigned integers.
///
/// The values are wrapped in FoR, without bit-packing, and its encoded child is split by
/// [`elias_fano_encode`]. The bucket counts are forced into [`Unary`], which gives Elias-Fano its
/// space bound, and the low bits are bit-packed.
///
/// Elias-Fano is in no file edition, so the compressor drops this scheme unless its encodings are
/// allowed. It is not in the default [`CompressionSession`](crate::CompressionSession) either.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct EliasFanoScheme;

impl Scheme for EliasFanoScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.elias_fano"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_unsigned_int() && !canonical.dtype().is_nullable()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        let mut encodings = vec![for_v1_id(), EliasFano.id(), Unary.id()];
        encodings.extend(BITPACKING_V1.produced_encodings());
        encodings
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // The split `elias_fano_encode` makes once FoR has removed the minimum.
        let len = data.array_len() as u64;
        let range = data.integer_stats(exec_ctx).erased().max_minus_min();
        let lower_bits = (range / len).checked_ilog2().unwrap_or(0);
        let buckets = (range >> lower_bits) + 1;
        // `unary_encode` refuses counts that average more than 63 values per bucket, which
        // run-length and dictionary encodings suit better.
        if buckets.saturating_mul(63) < len {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        // `len + buckets` upper bits, and `lower_bits` lower bits per value, which BitPacked pads
        // to whole `FL_CHUNK_SIZE` chunks.
        let padded_len = data.array_len().next_multiple_of(FL_CHUNK_SIZE);
        let bits_per_value = 1.0
            + buckets as f64 / len as f64
            + padded_len as f64 / len as f64 * f64::from(lower_bits);
        let ratio = data.array_as_primitive().ptype().bit_width() as f64 / bits_per_value;

        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            move |_compressor, data, best_so_far, _ctx, exec_ctx| {
                // The sorted check reads every value, so it runs only when EF could win.
                let threshold = best_so_far.and_then(EstimateScore::finite_ratio);
                if threshold.is_some_and(|t| ratio <= t) || !is_sorted(data.array(), exec_ctx)? {
                    return Ok(EstimateVerdict::Skip);
                }
                Ok(EstimateVerdict::Ratio(ratio))
            },
        )))
    }

    fn compress(
        &self,
        compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        let primitive = data.array().clone().execute::<PrimitiveArray>(exec_ctx)?;
        let for_array = FoR::encode(primitive, exec_ctx)?;
        let FoRReferences::Global(reference) = for_array.references() else {
            vortex_bail!("FoR::encode produced per-chunk references");
        };
        let biased = for_array
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?;
        let ef = elias_fano_encode(biased.as_view(), exec_ctx)?
            .ok_or_else(|| vortex_err!("cannot Elias-Fano encode an unsorted array"))?;

        let counts = ef.upper().clone().execute::<PrimitiveArray>(exec_ctx)?;
        let upper = unary_encode(counts.as_view(), exec_ctx)?
            .ok_or_else(|| vortex_err!("Elias-Fano bucket counts do not fit Unary"))?;
        let lows = ArrayAndStats::new(ef.lower().clone(), compress_ctx.merged_stats_options());
        let lower = BITPACKING_V1.compress(compressor, &lows, compress_ctx, exec_ctx)?;

        let ef = EliasFano::try_new(upper.into_array(), lower, ef.lower_bits())?;
        Ok(FoR::try_new(ef.into_array(), reference.clone())?.into_array())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_elias_fano::EliasFano;
    use vortex_elias_fano::EliasFanoArraySlotsExt;
    use vortex_elias_fano::Unary;
    use vortex_error::VortexResult;
    use vortex_fastlanes::BitPacked;
    use vortex_fastlanes::FoR;
    use vortex_fastlanes::FoRArraySlotsExt;

    use super::EliasFanoScheme;
    use crate::BtrBlocksCompressorBuilder;
    use crate::SESSION;

    #[test]
    fn test_compress() -> VortexResult<()> {
        let compressor = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme(&EliasFanoScheme)
            .build();
        let mut ctx = SESSION.create_execution_ctx();

        // Sorted, with a minimum for FoR to remove.
        let sorted =
            PrimitiveArray::from_iter((0..4096u64).map(|i| 1_000_000 + i * i)).into_array();
        let compressed = compressor.compress(&sorted, &mut ctx)?;
        assert_arrays_eq!(compressed, sorted, &mut ctx);
        let for_ = compressed.as_::<FoR>();
        let ef = for_.encoded().as_::<EliasFano>();
        assert!(ef.upper().is::<Unary>());
        assert!(ef.lower().is::<BitPacked>());

        // The same values in reverse are not sorted, so the scheme skips them.
        let reversed =
            PrimitiveArray::from_iter((0..4096u64).rev().map(|i| 1_000_000 + i * i)).into_array();
        assert!(!compressor.compress(&reversed, &mut ctx)?.is::<FoR>());

        // Sorted, but with too many values per bucket for Unary, so the scheme skips them.
        let duplicates = PrimitiveArray::from_iter((0..4096u64).map(|i| i / 410)).into_array();
        assert!(!compressor.compress(&duplicates, &mut ctx)?.is::<FoR>());
        Ok(())
    }
}
