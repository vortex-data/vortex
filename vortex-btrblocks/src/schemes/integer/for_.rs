// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Frame of Reference integer encoding.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Patched;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::patched::PatchedArraySlotsExt;
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
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_fastlanes::BitPacked;
use vortex_fastlanes::BitPackedArrayExt;
use vortex_fastlanes::FL_CHUNK_SIZE;
use vortex_fastlanes::FoR;
use vortex_fastlanes::FoRArrayExt;
use vortex_fastlanes::FoRArraySlotsExt;
use vortex_fastlanes::FoRReferences;
use vortex_fastlanes::for_v1_id;
use vortex_fastlanes::for_v2_id;

use super::BitPackingScheme;
use super::RunEndScheme;
use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::boolean::BoolRunEndScheme;

/// Number of whole 1,024-element FastLanes blocks sampled when evaluating blocked `FoR.v2`.
const ESTIMATE_BLOCKS: u32 = 8;

/// Minimum relative savings (5%) for blocked `FoR.v2` over `min(BitPacked.v1, FoR.v1)`.
const MIN_V2_RELATIVE_SAVINGS: f64 = 0.05;

/// Minimum absolute savings (0.4 bits/value) for blocked `FoR.v2` over `min(BitPacked.v1, FoR.v1)`.
const MIN_V2_BITS_SAVED: f64 = 0.4;

/// Minimum compression ratio (after patch-density penalty) for blocked `FoR.v2`.
const MIN_V2_COMPRESSION_RATIO: f64 = 1.05;

/// Access-cost factor penalizing blocked `FoR.v2` on 1-byte integers.
const EIGHT_BIT_ACCESS_COST_FACTOR: f64 = 1.12;

/// Nonlinear patch-density penalty weight (bits) ported from PR #9544.
const PATCH_DENSITY_COST_BITS: u64 = 320;

#[derive(Debug, Copy, Clone, PartialEq, Eq)]
enum FoRSchemeMode {
    V1,
    V2,
}

/// The FoR scheme that only produces single-reference arrays.
pub(crate) static FOR_V1: FoRScheme = FoRScheme::v1();

/// The FoR scheme that may produce one reference per 1024-element chunk.
pub(crate) static FOR_V2: FoRScheme = FoRScheme::v2();

/// Frame of Reference encoding.
///
/// The v1 mode subtracts one reference, the array's minimum. The v2 mode subtracts one reference
/// per 1024-element chunk, the chunk's minimum. Arrays with a single reference, including v2
/// arrays whose references compress to a constant, serialize as `fastlanes.for`, while per-chunk
/// references serialize as `fastlanes.for.v2`.
///
/// The encoded child is always bit-packed by [`BitPackingScheme`] called directly, not chosen by
/// the compressor, so this scheme declares the child's serialized IDs as its own. Excluding the
/// BitPacking scheme does not stop FoR from bit-packing; disallowing `fastlanes.bitpacked` does.
///
/// The default uses v1. [`refine`](Scheme::refine) picks v2 references when `fastlanes.for.v2` is
/// allowed.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FoRScheme {
    mode: FoRSchemeMode,
}

impl FoRScheme {
    /// Creates a FoR scheme configured for v1, which uses a single reference.
    pub const fn v1() -> Self {
        Self {
            mode: FoRSchemeMode::V1,
        }
    }

    /// Creates a FoR scheme configured for v2, which may use one reference per chunk.
    pub const fn v2() -> Self {
        Self {
            mode: FoRSchemeMode::V2,
        }
    }
}

impl Default for FoRScheme {
    fn default() -> Self {
        Self::v1()
    }
}

impl Scheme for FoRScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.int.for"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_int()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        let mut encodings = match self.mode {
            FoRSchemeMode::V1 => vec![for_v1_id()],
            FoRSchemeMode::V2 => vec![for_v1_id(), for_v2_id()],
        };
        encodings.extend(BitPackingScheme.produced_encodings());
        encodings
    }

    fn refine(&self, allowed: &dyn Fn(&ArrayId) -> bool) -> &dyn Scheme {
        if allowed(&for_v2_id()) {
            &FOR_V2
        } else {
            &FOR_V1
        }
    }

    /// Children: references=0 in v2 mode. The encoded child is always bit-packed.
    fn num_children(&self) -> usize {
        match self.mode {
            FoRSchemeMode::V1 => 0,
            FoRSchemeMode::V2 => 1,
        }
    }

    /// Dict codes always start at 0, so FoR (which subtracts the min) is a no-op.
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
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        // FoR only subtracts the min. Without further compression (e.g. BitPacking), the output is
        // the same size.
        if compress_ctx.finished_cascading() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let v1_est = v1_compression_ratio(data, exec_ctx);

        // Single-reference mode, nested 16x64 samples, single-block arrays, and RunEnd ends
        // (where random binary search probes regress severely under per-chunk indirection) stay
        // strictly on the analytical single-reference FoR.v1 estimate.
        if self.mode == FoRSchemeMode::V1
            || compress_ctx.is_sample()
            || data.array_len() <= FL_CHUNK_SIZE
            || is_runend_ends(&compress_ctx)
        {
            return CompressionEstimate::Verdict(
                v1_est.map_or(EstimateVerdict::Skip, EstimateVerdict::Ratio),
            );
        }

        let scheme = *self;
        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            move |compressor, data, _best_so_far, compress_ctx, exec_ctx| {
                let v2_est =
                    evaluate_v2_on_sample(scheme, compressor, data, &compress_ctx, exec_ctx)?;
                // Never underbid FoR.v1 on ties: reporting `max(v1_est, v2_est)` ensures FoR
                // does not lose to Dict/RunEnd when v2 fails its margin and falls back to v1.
                let verdict = match (v1_est, v2_est) {
                    (Some(v1), Some(v2)) => EstimateVerdict::Ratio(v1.max(v2)),
                    (Some(v1), None) => EstimateVerdict::Ratio(v1),
                    (None, Some(v2)) => EstimateVerdict::Ratio(v2),
                    (None, None) => EstimateVerdict::Skip,
                };
                Ok(verdict)
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
        let use_v2 = self.mode == FoRSchemeMode::V2
            && !compress_ctx.is_sample()
            && data.array_len() > FL_CHUNK_SIZE
            && !is_runend_ends(&compress_ctx)
            && evaluate_v2_on_sample(*self, compressor, data, &compress_ctx, exec_ctx)?.is_some();

        let for_array = if use_v2 {
            FoR::encode_chunked(primitive, exec_ctx)?
        } else {
            FoR::encode(primitive, exec_ctx)?
        };
        let biased = for_array
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?;

        // Immediately bitpack when residuals satisfy BitPacking's non-negative precondition.
        // Signed wrapping subtraction on a block wider than `T::MAX` can produce negative
        // residuals; in that case fall back to selecting another encoding (excluding FoR)
        // rather than materializing an uncompressed FoR array larger than the input.
        let Some(compressed) = try_bitpack_residuals(compressor, biased, &compress_ctx, exec_ctx)?
        else {
            return compressor.compress_child(data.array(), &compress_ctx, self.id(), 0, exec_ctx);
        };

        // TODO(connor): This should really be `new_unchecked`.
        let for_compressed = match for_array.references() {
            FoRReferences::Global(reference) => FoR::try_new(compressed, reference.clone())?,
            FoRReferences::Blocked(references) => {
                let references =
                    compressor.compress_child(references, &compress_ctx, self.id(), 0, exec_ctx)?;
                FoR::try_new_chunked(compressed, references, for_array.offset())?
            }
        };
        for_compressed
            .as_ref()
            .statistics()
            .inherit_from(for_array.as_ref().statistics());

        Ok(for_compressed.into_array())
    }
}

/// Returns true when compressing the `ends` child of a RunEnd array, where binary search
/// `scalar_at` probes would incur extra per-block reference lookups under `FoR.v2`.
fn is_runend_ends(compress_ctx: &CompressorContext) -> bool {
    compress_ctx.cascade_history().iter().any(|&(id, child)| {
        (id == RunEndScheme.id() && child == 1) || (id == BoolRunEndScheme.id() && child == 0)
    })
}

/// Analytical compression ratio for single-reference `FoR.v1`, or `None` when v1 does not pay off.
fn v1_compression_ratio(data: &ArrayAndStats, exec_ctx: &mut ExecutionCtx) -> Option<f64> {
    let stats = data.integer_stats(exec_ctx);

    // Only apply when the min is not already zero.
    if stats.erased().min_is_zero() {
        return None;
    }

    // Difference between max and min.
    let for_bitwidth = stats.erased().max_minus_min().checked_ilog2()? + 1;

    // If BitPacking can be applied (only non-negative values) and FoR doesn't reduce bit width
    // compared to BitPacking, don't use FoR since it has a small amount of overhead (storing
    // the reference) for effectively no benefits.
    if let Some(max_log) = stats
        .erased()
        .max_ilog2()
        // Only skip FoR when min >= 0, otherwise BitPacking can't be applied without ZigZag.
        .filter(|_| !stats.erased().min_is_negative())
    {
        let bitpack_bitwidth = max_log + 1;
        if for_bitwidth >= bitpack_bitwidth {
            return None;
        }
    }

    let full_width: u32 = data
        .array_as_primitive()
        .ptype()
        .bit_width()
        .try_into()
        .vortex_expect("bit width must fit in u32");

    Some(full_width as f64 / for_bitwidth as f64)
}

/// Nonlinear patch-density penalty in bytes: quadratic in patch count to discourage high-patch
/// blocked encodings that inflate random-access and take cost.
fn patch_density_cost_bytes(len: usize, patch_count: usize) -> u64 {
    if len == 0 {
        return 0;
    }
    let patch_count = u64::try_from(patch_count).unwrap_or(u64::MAX);
    let len = u64::try_from(len).unwrap_or(u64::MAX);
    patch_count
        .saturating_mul(patch_count)
        .saturating_mul(PATCH_DENSITY_COST_BITS)
        .div_ceil(len.saturating_mul(8))
}

fn extract_patch_count(array: &ArrayRef) -> usize {
    if let Some(bp) = array.as_opt::<BitPacked>() {
        bp.patches().map_or(0, |p| p.indices().len())
    } else if let Some(patched) = array.as_opt::<Patched>() {
        patched.patch_indices().len()
    } else {
        0
    }
}

fn beats_v2_margin(candidate: u64, baseline: u64, sample_len: usize) -> bool {
    if candidate >= baseline || baseline == 0 || sample_len == 0 {
        return false;
    }
    let saved = (baseline - candidate) as f64;
    let rel_savings = saved / baseline as f64;
    let bits_saved = (saved * 8.0) / sample_len as f64;
    rel_savings >= MIN_V2_RELATIVE_SAVINGS && bits_saved >= MIN_V2_BITS_SAVED
}

/// Returns true when signed `residuals` contain any negative non-null value.
/// Unsigned residuals cannot be negative and skip the check without computing stats.
fn has_negative_residuals(residuals: &PrimitiveArray, exec_ctx: &mut ExecutionCtx) -> bool {
    if !residuals.ptype().is_signed_int() {
        return false;
    }
    vortex_array::match_each_signed_integer_ptype!(residuals.ptype(), |T| {
        has_negative_typed::<T>(residuals, exec_ctx)
    })
}

fn has_negative_typed<T>(residuals: &PrimitiveArray, exec_ctx: &mut ExecutionCtx) -> bool
where
    T: vortex_array::dtype::NativePType
        + PartialOrd
        + Default
        + for<'a> TryFrom<&'a vortex_array::scalar::Scalar, Error = vortex_error::VortexError>,
{
    residuals
        .as_ref()
        .statistics()
        .compute_min::<T>(exec_ctx)
        .is_some_and(|min| min < T::default())
}

/// Bit-packs `residuals` when they satisfy BitPacking's non-negative precondition; returns `None`
/// when signed wrapping subtraction produced negative residuals.
fn try_bitpack_residuals(
    compressor: &CascadingCompressor,
    residuals: PrimitiveArray,
    compress_ctx: &CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<Option<ArrayRef>> {
    if has_negative_residuals(&residuals, exec_ctx) {
        return Ok(None);
    }
    let data = ArrayAndStats::new(residuals.into_array(), Default::default());
    BitPackingScheme
        .compress(compressor, &data, compress_ctx.clone(), exec_ctx)
        .map(Some)
}

/// Evaluates blocked `FoR.v2` against `min(BitPacked.v1, FoR.v1)` on up to 8 aligned
/// 1,024-element blocks, returning the adjusted compression ratio when `FoR.v2` clears all
/// guardrails.
fn evaluate_v2_on_sample(
    scheme: FoRScheme,
    compressor: &CascadingCompressor,
    data: &ArrayAndStats,
    compress_ctx: &CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<Option<f64>> {
    let sample = sample_aligned_chunks(data.array(), FL_CHUNK_SIZE, ESTIMATE_BLOCKS)
        .unwrap_or_else(|| data.array().clone())
        .execute::<PrimitiveArray>(exec_ctx)?;
    let before_nbytes = sample.nbytes();
    let sample_len = sample.len();
    if sample_len <= FL_CHUNK_SIZE || before_nbytes == 0 {
        return Ok(None);
    }
    let byte_width = sample.ptype().byte_width();

    let for_v2 = FoR::encode_chunked(sample.clone(), exec_ctx)?;
    let FoRReferences::Blocked(references) = for_v2.references() else {
        return Ok(None);
    };

    let biased_v2 = for_v2
        .encoded()
        .clone()
        .execute::<PrimitiveArray>(exec_ctx)?;
    let Some(packed_v2) = try_bitpack_residuals(compressor, biased_v2, compress_ctx, exec_ctx)?
    else {
        return Ok(None);
    };
    let v2_patches = extract_patch_count(&packed_v2);
    let compressed_refs =
        compressor.compress_child(references, compress_ctx, scheme.id(), 0, exec_ctx)?;
    let for_v2_compressed = FoR::try_new_chunked(packed_v2, compressed_refs, for_v2.offset())?;

    let v2_raw = for_v2_compressed.nbytes();
    let v2_penalized = v2_raw.saturating_add(patch_density_cost_bytes(sample_len, v2_patches));

    // Compute exact baseline `min(BitPacked.v1, FoR.v1)` on the same aligned sample.
    let sample_data = ArrayAndStats::new(
        sample.clone().into_array(),
        compress_ctx.merged_stats_options(),
    );
    let sample_stats = sample_data.integer_stats(exec_ctx);
    let mut baseline_raw = before_nbytes;
    let mut baseline_penalized = before_nbytes;

    if !sample_stats.erased().min_is_negative() {
        let bp =
            BitPackingScheme.compress(compressor, &sample_data, compress_ctx.clone(), exec_ctx)?;
        let bp_raw = bp.nbytes();
        let bp_penalized = bp_raw.saturating_add(patch_density_cost_bytes(
            sample_len,
            extract_patch_count(&bp),
        ));
        baseline_raw = baseline_raw.min(bp_raw);
        baseline_penalized = baseline_penalized.min(bp_penalized);
    }

    if !sample_stats.erased().min_is_zero()
        && let Ok(for_v1) = FoR::encode(sample, exec_ctx)
    {
        let biased_v1 = for_v1
            .encoded()
            .clone()
            .execute::<PrimitiveArray>(exec_ctx)?;
        if let Some(packed_v1) =
            try_bitpack_residuals(compressor, biased_v1, compress_ctx, exec_ctx)?
        {
            let v1_patches = extract_patch_count(&packed_v1);
            let for_v1_compressed = match for_v1.references() {
                FoRReferences::Global(reference) => FoR::try_new(packed_v1, reference.clone())?,
                FoRReferences::Blocked(_) => {
                    unreachable!("FoR::encode produces a global reference")
                }
            };
            let v1_raw = for_v1_compressed.nbytes();
            let v1_penalized =
                v1_raw.saturating_add(patch_density_cost_bytes(sample_len, v1_patches));
            baseline_raw = baseline_raw.min(v1_raw);
            baseline_penalized = baseline_penalized.min(v1_penalized);
        }
    }

    if !beats_v2_margin(v2_raw, baseline_raw, sample_len)
        || !beats_v2_margin(v2_penalized, baseline_penalized, sample_len)
    {
        return Ok(None);
    }

    let ratio = before_nbytes as f64 / v2_penalized.max(1) as f64;
    if ratio < MIN_V2_COMPRESSION_RATIO {
        return Ok(None);
    }

    let adjusted = if byte_width == 1 {
        ratio / EIGHT_BIT_ACCESS_COST_FACTOR
    } else {
        ratio
    };
    Ok(Some(adjusted))
}

#[cfg(test)]
mod tests {
    use rstest::rstest;
    use vortex_array::ArrayId;
    use vortex_fastlanes::for_v1_id;
    use vortex_fastlanes::for_v2_id;

    use super::BitPackingScheme;
    use super::FOR_V1;
    use super::FOR_V2;
    use crate::Scheme;
    use crate::SchemeExt;

    /// Both variants refine to v2 exactly when the v2 ID is allowed.
    #[rstest]
    #[case::neither(false, false, false)]
    #[case::v1(true, false, false)]
    #[case::v2_only(false, true, true)]
    #[case::both(true, true, true)]
    fn refine_picks_v2_when_v2_id_is_allowed(
        #[case] allow_v1: bool,
        #[case] allow_v2: bool,
        #[case] expect_v2: bool,
        #[values(&FOR_V1, &FOR_V2)] scheme: &'static dyn Scheme,
    ) {
        let allowed =
            |id: &ArrayId| (allow_v1 && *id == for_v1_id()) || (allow_v2 && *id == for_v2_id());
        let refined = scheme.refine(&allowed);
        assert_eq!(refined.id(), scheme.id());
        let mut expected = if expect_v2 {
            vec![for_v1_id(), for_v2_id()]
        } else {
            vec![for_v1_id()]
        };
        expected.extend(BitPackingScheme.produced_encodings());
        assert_eq!(refined.produced_encodings(), expected);
    }
}
