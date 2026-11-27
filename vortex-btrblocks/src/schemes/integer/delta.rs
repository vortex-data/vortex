// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FastLanes Delta integer encoding.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::arrays::PrimitiveArray;
use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::DecimalDictScheme;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::sample_aligned_chunks;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateScore;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexResult;
use vortex_fastlanes::Delta;
use vortex_fastlanes::FL_CHUNK_SIZE;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::Child;
use crate::CompressorContext;
use crate::Encoded;
use crate::GenerateStatsOptions;
use crate::Scheme;
use crate::SchemeExt;

/// FastLanes Delta encoding for smooth / near-monotone integers.
///
/// Delta replaces each value with its difference from an earlier value (at the FastLanes lane
/// stride), so a later cascade layer (FoR / BitPacking) packs the smaller residuals. It only
/// pays off when those residuals span meaningfully fewer bits than the values themselves.
///
/// The minimum penalized compression ratio required for Delta to be selected is configurable via
/// [`DeltaScheme::new`]; [`DeltaScheme::default`] uses a ratio of `1.25`.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct DeltaScheme {
    min_ratio: f64,
}

impl DeltaScheme {
    /// Creates a Delta scheme requiring `min_ratio` after the delta penalty before it wins.
    ///
    /// Pass a higher ratio to make Delta more conservative, or a lower one to select it more
    /// eagerly. [`DeltaScheme::default`] uses a ratio of `1.25`.
    pub const fn new(min_ratio: f64) -> Self {
        Self { min_ratio }
    }
}

impl Default for DeltaScheme {
    fn default() -> Self {
        Self::new(1.25)
    }
}

/// Multiplicative penalty applied to Delta's estimated compression ratio.
///
/// Unlike FoR/BitPacking, Delta breaks random access and adds a prefix-sum decode pass, and it
/// carries a structural sign bit on its residuals. We therefore require Delta to be meaningfully
/// (~5%) smaller than the best alternative before it wins, rather than picking it for a
/// single-bit gain. This factor encodes that "delta tax".
const DELTA_PENALTY: f64 = 0.95;

/// Minimum length before Delta is worth considering (one FastLanes chunk).
const MIN_DELTA_LEN: usize = 1024;

/// Number of whole FastLanes chunks Delta's estimate delta-encodes, instead of the whole array.
const SAMPLE_CHUNKS: u32 = 4;

/// Penalized compression ratio for `len` values of `full_width` bits whose transposed residuals
/// need `delta_bits` bits each.
///
/// Delta writes two children and both are sized by the padded length, since `delta_compress`
/// rounds the input up to whole [`FL_CHUNK_SIZE`] chunks. The deltas child holds `padded_len`
/// residuals of `delta_bits`; the bases child holds one base per lane per chunk, and a chunk of
/// `W`-bit values has `FL_CHUNK_SIZE / W` lanes, so its bases occupy [`FL_CHUNK_SIZE`] bits per
/// chunk whatever `W` is: one further bit per residual. Charging the bases at full width is an
/// upper bound, as the cascade compresses that child like any other.
///
/// `len` must be non-zero; callers gate on [`MIN_DELTA_LEN`].
fn penalized_ratio(len: usize, full_width: f64, delta_bits: f64) -> f64 {
    let padded_len = len.next_multiple_of(FL_CHUNK_SIZE) as f64;
    (len as f64) * full_width / (padded_len * (delta_bits + 1.0)) * DELTA_PENALTY
}

impl Scheme for DeltaScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.delta"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![Delta.id()]
    }

    fn num_children(&self) -> usize {
        2
    }

    /// Delta-encode the data at most once per path: exclude Delta from the subtrees of both the
    /// bases and the deltas children so we never delta-encode data that was already delta-encoded.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![DescendantExclusion {
            excluded: self.id(),
            children: ChildSelection::All,
        }]
    }

    /// Delta over dictionary codes just adds indirection: codes are compact integers with no
    /// monotone structure, so (like FoR/Sequence) skip the codes child.
    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        vec![
            AncestorExclusion {
                ancestor: IntDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: FloatDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: StringDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: BinaryDictScheme.id(),
                children: ChildSelection::One(1),
            },
            AncestorExclusion {
                ancestor: DecimalDictScheme.id(),
                children: ChildSelection::One(1),
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // Delta only pays off if a later cascade layer (FoR/BitPacking) packs the residuals.
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        // Too short to transpose into FastLanes chunks meaningfully.
        if data.array_len() < MIN_DELTA_LEN {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Estimating Delta needs the real transposed-delta span, so defer to a callback that
        // delta-encodes a sample of whole FastLanes chunks and measures the residual range.
        let min_ratio = self.min_ratio;
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            move |_compressor, data, best_so_far, ctx, exec_ctx| {
                let full_width = data.array().dtype().as_ptype().bit_width() as f64;
                let len = data.array_len();

                // Delta's best case is constant residuals (0 delta bits, paying only for bases).
                // If even that, after the penalty, can't beat the incumbent, skip before doing the
                // encode work.
                let threshold = best_so_far.and_then(EstimateScore::finite_ratio);
                if threshold.is_some_and(|t| penalized_ratio(len, full_width, 0.0) <= t) {
                    return Ok(EstimateVerdict::Skip);
                }

                // Delta encodes each FastLanes chunk on its own, with fresh bases, so whole
                // aligned chunks have the same residuals as those rows in the full encode (except
                // where leading nulls are filled from the previous chunk). A sample can miss rare
                // outliers, such as a series restarting, but the cascade under Delta patches
                // those rather than widening every residual for them. Inside another scheme's
                // sample the array is already small, so measure it whole.
                let sampled = if ctx.is_sample() {
                    None
                } else {
                    sample_aligned_chunks(data.array(), FL_CHUNK_SIZE, SAMPLE_CHUNKS)
                };
                let primitive = sampled
                    .unwrap_or_else(|| data.array().clone())
                    .execute::<PrimitiveArray>(exec_ctx)?;

                // Measure the actual FastLanes transposed-delta span. This is the lane-stride
                // difference that gets bit-packed, not the lag-1 difference (which the transpose
                // makes optimistic), so it is what truly drives the compressed size.
                let (_bases, deltas) = vortex_fastlanes::delta_compress(&primitive, exec_ctx)?;
                let delta_stats =
                    ArrayAndStats::new(deltas.into_array(), GenerateStatsOptions::default());
                let span = delta_stats.integer_stats(exec_ctx).erased().max_minus_min();

                // Bits needed to FoR-pack the residuals. A zero span means all sampled residuals
                // are identical (e.g. piecewise-constant chunks), so `deltas` collapses to a
                // constant (0 delta bits) and only the bases child costs space.
                let delta_bits = match span.checked_ilog2() {
                    Some(l) => (l + 1) as f64,
                    None => 0.0,
                };

                let ratio = penalized_ratio(len, full_width, delta_bits);
                if ratio <= min_ratio {
                    return Ok(EstimateVerdict::Skip);
                }
                Ok(EstimateVerdict::Ratio(ratio))
            },
        )))
    }

    /// Children: bases=0, deltas=1. Delta stores the input's validity, which is registered.
    fn encode(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: &CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<Encoded> {
        let primitive = data.array().clone().execute::<PrimitiveArray>(exec_ctx)?;
        let len = primitive.len();
        let (bases, deltas) = vortex_fastlanes::delta_compress(&primitive, exec_ctx)?;

        Ok(Encoded::new(
            vec![
                Child::new(self, 0, bases.into_array()),
                Child::new(self, 1, deltas.into_array()),
            ],
            Vec::new(),
            move |children, _| {
                let [bases, deltas] = children.compressed()?;
                Delta::try_new(bases, deltas, children.validity()?, 0, len)
                    .map(IntoArray::into_array)
            },
        )
        .with_validity(primitive.validity()?))
    }

    #[expect(clippy::unimplemented, reason = "replaced by `encode`")]
    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        _data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        _exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        unimplemented!("{} compresses through Scheme::encode", self.scheme_name())
    }
}

#[cfg(test)]
mod tests {
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::assert_arrays_eq;
    use vortex_error::VortexResult;
    use vortex_fastlanes::Delta;
    use vortex_sequence::Sequence;

    use super::DeltaScheme;
    use crate::BtrBlocksCompressorBuilder;
    use crate::SESSION;
    use crate::schemes::integer::SequenceScheme;

    static DELTA: DeltaScheme = DeltaScheme::new(1.25);

    #[test]
    fn test_delta_compresses_zero_span_step_function() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input =
            PrimitiveArray::from_iter((0..65_536u64).map(|i| (i / 1024) * 1_000_000)).into_array();

        // With Delta and Sequence competing, Sequence skips (non-unique values) and Delta must
        // score `span == 0` with `delta_bits = 0` rather than skipping.
        let compressor = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme(&SequenceScheme)
            .with_new_scheme(&DELTA)
            .build();
        let compressed = compressor.compress(&input, &mut ctx)?;
        assert!(
            compressed.is::<Delta>(),
            "expected Delta, got {}",
            compressed.encoding_id()
        );
        // `deltas` collapses to a constant and `bases` holds 16 u64s per 1024-row chunk (~64x).
        assert!(compressed.nbytes() < input.nbytes() / 50);
        assert_arrays_eq!(compressed, input, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_delta_wins_full_compressor_on_zero_span_piecewise_linear() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        // Each 1024-row FastLanes chunk is a stride-1 ramp starting at a wide jump, so all 65,536
        // values are unique (RunEnd/RLE/Dict/Sparse skip) and non-linear globally (Sequence skips),
        // while every transposed delta inside a chunk is identical (`span == 0`).
        let input = PrimitiveArray::from_iter(
            (0..65_536u64).map(|i| (i / 1024) * (1u64 << 50) + (i % 1024)),
        )
        .into_array();

        let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .build();
        let compressed = compressor.compress(&input, &mut ctx)?;
        assert!(
            compressed.is::<Delta>(),
            "expected Delta, got {}",
            compressed.encoding_id()
        );
        assert!(compressed.nbytes() < input.nbytes() / 64);
        assert_arrays_eq!(compressed, input, &mut ctx);
        Ok(())
    }

    #[test]
    fn test_delta_zero_span_still_defers_to_sequence_on_linear_counter() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = PrimitiveArray::from_iter(0..65_536u64).into_array();

        let compressor = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .build();
        let compressed = compressor.compress(&input, &mut ctx)?;
        assert!(
            compressed.is::<Sequence>(),
            "expected Sequence, got {}",
            compressed.encoding_id()
        );
        assert_arrays_eq!(compressed, input, &mut ctx);
        Ok(())
    }
}
