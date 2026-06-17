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
use vortex_compressor::estimate::EstimateScore;
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
    /// Creates a Delta scheme requiring `min_ratio` (after the [`DELTA_PENALTY`]) before it wins.
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
            move |_compressor, data, best_so_far, _ctx, exec_ctx| {
                let primitive = data.array().clone().execute::<PrimitiveArray>(exec_ctx)?;
                let full_width = primitive.ptype().bit_width() as f64;

                // Delta's best case is residuals collapsing to a single bit. If even that, after
                // the penalty, can't beat the incumbent, skip before doing the encode work.
                let threshold = best_so_far.and_then(EstimateScore::finite_ratio);
                if threshold.is_some_and(|t| full_width * DELTA_PENALTY <= t) {
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
