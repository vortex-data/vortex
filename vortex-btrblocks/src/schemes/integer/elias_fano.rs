// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Elias-Fano integer encoding for monotonically non-decreasing sequences.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::VTable;
use vortex_array::aggregate_fn::fns::is_sorted::is_sorted;
use vortex_array::arrays::Constant;
use vortex_compressor::builtins::BinaryDictScheme;
use vortex_compressor::builtins::FloatDictScheme;
use vortex_compressor::builtins::IntDictScheme;
use vortex_compressor::builtins::StringDictScheme;
use vortex_compressor::scheme::AncestorExclusion;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::EstimateScore;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_elias_fano::EliasFano;
use vortex_elias_fano::elias_fano_encode;
use vortex_elias_fano::encoded_bit_size;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_fastlanes::BitPacked;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::string::FSSTScheme;

/// Elias-Fano encoding for monotonically non-decreasing integers.
///
/// Stores each value in about `log2(u / n) + 2` bits for `n` values over a universe of `u`, against
/// bit-packing's `ceil(log2(u))`. The saving is therefore roughly `log2(n)` bits per value and it
/// widens with row count, which makes the encoding most valuable on exactly the columns that are
/// largest: sorted keys, timestamps, and a list column's offsets.
///
/// The minimum penalized compression ratio required before Elias-Fano is selected is configurable
/// via [`EliasFanoScheme::new`]; [`EliasFanoScheme::default`] uses a ratio of `1.2`.
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct EliasFanoScheme {
    min_ratio: f64,
}

impl EliasFanoScheme {
    /// Creates an Elias-Fano scheme requiring `min_ratio` after the penalty before it wins.
    ///
    /// Pass a higher ratio to make Elias-Fano more conservative, or a lower one to select it more
    /// eagerly. [`EliasFanoScheme::default`] uses a ratio of `1.2`.
    pub const fn new(min_ratio: f64) -> Self {
        Self { min_ratio }
    }
}

impl Default for EliasFanoScheme {
    fn default() -> Self {
        Self::new(1.2)
    }
}

/// Multiplicative penalty applied to Elias-Fano's estimated compression ratio.
///
/// A point lookup costs a sampled `select1` — a pointer read plus a bounded popcount scan — where
/// bit-packing costs one unpack. Elias-Fano keeps random access, unlike Delta, so the tax is no
/// heavier than Delta's; but it is not free either, so we require a real size win rather than
/// picking Elias-Fano for a single-bit gain.
const ELIAS_FANO_PENALTY: f64 = 0.95;

/// Minimum length before Elias-Fano is worth considering.
///
/// Below one FastLanes block the padding of the low-bits child, the two sample tables and the guard
/// bits dominate, and the asymptotic saving has not arrived yet.
const MIN_ELIAS_FANO_LEN: usize = 1024;

impl Scheme for EliasFanoScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.elias_fano"
    }

    /// Elias-Fano has nowhere to put a null: a null has no position in an ordering, and the array
    /// reports `Validity::NonNullable` unconditionally. A nullable dtype can therefore never be
    /// encoded, whatever its null count, so it is rejected here rather than in the estimate.
    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int() && !canonical.dtype().is_nullable()
    }

    /// The low-bits child is built inside `elias_fano_encode` rather than handed back through
    /// `compress_child`, so it never passes the edition filter on its own. Declaring both of the
    /// encodings that encoder can produce keeps `retain_allowed_encodings` honest.
    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![EliasFano.id(), BitPacked.id(), Constant.id()]
    }

    /// Two different reasons to decline, both about the parent's access pattern rather than the
    /// data:
    ///
    /// - **Dictionary codes** have no monotone structure, so Elias-Fano over them just adds
    ///   indirection. Same exclusion FoR, Sequence and Delta declare.
    /// - **FSST's children** are always fully materialised, never read through a pushdown: the
    ///   `like` kernel calls `codes.offsets().execute::<PrimitiveArray>()` and canonicalisation
    ///   reads `uncompressed_lengths` as a slice. Measured on this branch, an Elias-Fano bulk decode
    ///   costs about 9x a bit-packed unpack of the same values (172 µs vs 19 µs for 65,536 u64), so
    ///   trading roughly 3x space for that on a child every string operation decodes is the wrong
    ///   way round. The cost model prices space only; a scalar penalty cannot express a 9x decode
    ///   difference, so it is excluded structurally instead.
    fn ancestor_exclusions(&self) -> Vec<AncestorExclusion> {
        vec![
            AncestorExclusion {
                ancestor: FSSTScheme.id(),
                children: ChildSelection::All,
            },
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
        // A contiguous sample of a sorted column keeps the full span but a fraction of the rows, so
        // `log2(span / n)` comes out too wide and the estimate is biased against Elias-Fano. Price
        // the whole array instead, as SequenceScheme does for the same reason.
        if compress_ctx.is_sample() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }
        if data.array_len() < MIN_ELIAS_FANO_LEN {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let min_ratio = self.min_ratio;
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            move |_compressor, data, best_so_far, _ctx, exec_ctx| {
                let primitive = data.array_as_primitive();
                let full_width = primitive.ptype().bit_width() as f64;
                let n = data.array_len();

                let stats = data.integer_stats(exec_ctx);
                if stats.null_count() > 0 {
                    return Ok(EstimateVerdict::Skip);
                }

                // On a non-decreasing sequence the minimum is the first element, which is exactly
                // the reference the encoder subtracts, so this span is the one it will see. That
                // holds for signed types too: both sides work in wrapping two's complement.
                let span = stats.erased().max_minus_min();

                // The cost model is exact and costs no allocation, so price the candidate before
                // paying for the O(n) sortedness scan below.
                let ratio = (n as f64 * full_width) / encoded_bit_size(span, n)? as f64
                    * ELIAS_FANO_PENALTY;
                if ratio <= min_ratio {
                    return Ok(EstimateVerdict::Skip);
                }
                let threshold = best_so_far.and_then(EstimateScore::finite_ratio);
                if threshold.is_some_and(|t| ratio <= t) {
                    return Ok(EstimateVerdict::Skip);
                }

                // Last, because it is the only part that reads every value. The result is cached as
                // `Stat::IsSorted`, which `ListArray::try_new` then reuses rather than rescanning.
                if !is_sorted(data.array(), exec_ctx)? {
                    return Ok(EstimateVerdict::Skip);
                }

                Ok(EstimateVerdict::Ratio(ratio))
            },
        )))
    }

    fn compress(
        &self,
        _compressor: &CascadingCompressor,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> VortexResult<ArrayRef> {
        if data.integer_stats(exec_ctx).null_count() > 0 {
            vortex_bail!("Elias-Fano encoding does not support nulls");
        }
        elias_fano_encode(data.array_as_primitive(), exec_ctx).map(IntoArray::into_array)
    }
}
