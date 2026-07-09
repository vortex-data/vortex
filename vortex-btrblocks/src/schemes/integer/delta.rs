// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! FastLanes Delta integer encoding.

use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::PrimitiveArray;
use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::estimate::CompressionEstimate;
use vortex_compressor::estimate::DeferredEstimate;
use vortex_compressor::estimate::EstimateVerdict;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_error::VortexResult;
use vortex_fastlanes::Delta;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
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

impl Scheme for DeltaScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.delta"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
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
        // delta-encodes the array and measures the residual range.
        let min_ratio = self.min_ratio;
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            move |_compressor, data, threshold, _ctx, exec_ctx| {
                let primitive = data.array().clone().execute::<PrimitiveArray>(exec_ctx)?;
                let full_width = primitive.ptype().bit_width() as f64;

                // Delta's best case is residuals collapsing to a single bit. If even that, after
                // the penalty, can't beat the incumbent, skip before doing the encode work.
                if threshold.best_case_ratio_cannot_win(full_width * DELTA_PENALTY) {
                    return Ok(EstimateVerdict::Skip);
                }

                // Measure the actual FastLanes transposed-delta span. This is the lane-stride
                // difference that gets bit-packed, not the lag-1 difference (which the transpose
                // makes optimistic), so it is what truly drives the compressed size.
                let (_bases, deltas) = vortex_fastlanes::delta_compress(&primitive, exec_ctx)?;
                let delta_stats =
                    ArrayAndStats::new(deltas.into_array(), GenerateStatsOptions::default());
                let span = delta_stats.integer_stats(exec_ctx).erased().max_minus_min();

                // Bits needed to FoR-pack the residuals. A zero span means constant deltas, which
                // SequenceScheme already captures more cheaply, so defer to it.
                let delta_bits = match span.checked_ilog2() {
                    Some(l) => (l + 1) as f64,
                    None => return Ok(EstimateVerdict::Skip),
                };

                let ratio = full_width / delta_bits * DELTA_PENALTY;
                if ratio <= min_ratio {
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
        let len = primitive.len();
        let (bases, deltas) = vortex_fastlanes::delta_compress(&primitive, exec_ctx)?;

        let compressed_bases = compressor.compress_child(
            &bases.into_array(),
            &compress_ctx,
            self.id(),
            0,
            exec_ctx,
        )?;
        let compressed_deltas = compressor.compress_child(
            &deltas.into_array(),
            &compress_ctx,
            self.id(),
            1,
            exec_ctx,
        )?;

        Delta::try_new(compressed_bases, compressed_deltas, 0, len).map(IntoArray::into_array)
    }
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use rand::RngExt;
    use rand::SeedableRng;
    use rand::rngs::StdRng;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::Constant;
    use vortex_array::arrays::ConstantArray;
    use vortex_array::dtype::Nullability;
    use vortex_array::scalar::Scalar;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_session::VortexSession;

    use super::*;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(vortex_array::array_session);

    /// Immediate fixed-ratio competitor; its `compress` emits a tiny constant array so the
    /// winner is observable from the output encoding.
    #[derive(Debug)]
    struct FixedRatioScheme {
        ratio: f64,
    }

    impl Scheme for FixedRatioScheme {
        fn scheme_name(&self) -> &'static str {
            "test.fixed_ratio"
        }

        fn matches(&self, canonical: &Canonical) -> bool {
            canonical.dtype().is_int()
        }

        fn expected_compression_ratio(
            &self,
            _data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> CompressionEstimate {
            CompressionEstimate::Verdict(EstimateVerdict::Ratio(self.ratio))
        }

        fn compress(
            &self,
            _compressor: &CascadingCompressor,
            data: &ArrayAndStats,
            _compress_ctx: CompressorContext,
            _exec_ctx: &mut ExecutionCtx,
        ) -> VortexResult<ArrayRef> {
            Ok(ConstantArray::new(
                Scalar::primitive(0u64, Nullability::NonNullable),
                data.array_len(),
            )
            .into_array())
        }
    }

    /// Near-monotone u64 data: Delta's residual span is a few bits, so its real penalized
    /// ratio is comfortably above 2.0 but nowhere near its 64-bit best case.
    fn monotone_jitter_u64() -> ArrayRef {
        let mut rng = StdRng::seed_from_u64(42);
        let mut value = 1_700_000_000_000u64;
        let values: Buffer<u64> = (0..4096)
            .map(|_| {
                value += 900 + rng.random_range(0..200);
                value
            })
            .collect();
        PrimitiveArray::new(values, Validity::NonNullable).into_array()
    }

    /// With the incumbent below Delta's achievable ratio, the callback must proceed past
    /// the best-case threshold check and win with its measured estimate.
    #[test]
    fn delta_wins_over_low_threshold() -> VortexResult<()> {
        static COMPETITOR: FixedRatioScheme = FixedRatioScheme { ratio: 2.0 };
        static DELTA: DeltaScheme = DeltaScheme::new(1.25);
        let compressor = CascadingCompressor::new(vec![&COMPETITOR, &DELTA]);

        let mut exec_ctx = SESSION.create_execution_ctx();
        let compressed = compressor.compress(&monotone_jitter_u64(), &mut exec_ctx)?;

        assert!(compressed.is::<Delta>());
        Ok(())
    }

    /// With the incumbent at exactly Delta's best case (`full_width * DELTA_PENALTY`, the
    /// same expression the callback computes), Delta must not be chosen — the same decision
    /// the pre-threshold-handle `max_ratio <= best` skip produced on this input.
    #[test]
    fn delta_loses_at_best_case_tie() -> VortexResult<()> {
        static COMPETITOR: FixedRatioScheme = FixedRatioScheme {
            ratio: 64.0 * DELTA_PENALTY,
        };
        static DELTA: DeltaScheme = DeltaScheme::new(1.25);
        let compressor = CascadingCompressor::new(vec![&COMPETITOR, &DELTA]);

        let mut exec_ctx = SESSION.create_execution_ctx();
        let compressed = compressor.compress(&monotone_jitter_u64(), &mut exec_ctx)?;

        assert!(compressed.is::<Constant>());
        Ok(())
    }

    /// Wide random residuals put Delta's penalized ratio below its `min_ratio`, so with no
    /// competitor the array must stay canonical.
    #[test]
    fn delta_skips_below_min_ratio() -> VortexResult<()> {
        static DELTA: DeltaScheme = DeltaScheme::new(1.25);
        let compressor = CascadingCompressor::new(vec![&DELTA]);

        let mut rng = StdRng::seed_from_u64(43);
        let values: Buffer<u64> = (0..4096).map(|_| rng.random::<u64>()).collect();
        let array = PrimitiveArray::new(values, Validity::NonNullable).into_array();

        let mut exec_ctx = SESSION.create_execution_ctx();
        let compressed = compressor.compress(&array, &mut exec_ctx)?;

        assert!(!compressed.is::<Delta>());
        Ok(())
    }
}
