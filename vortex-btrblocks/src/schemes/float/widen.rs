// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Lossless floating-point width-narrowing scheme backed by [`WidenArray`].
//!
//! Narrowing an `f64` column whose valid values all round-trip bit-for-bit through `f32` (or
//! narrowing `f64`/`f32` through `f16`) halves or quarters raw storage and lets child float
//! schemes such as `ALP` and `ALPRD` operate at the narrower width without losing a single bit.
//!
//! Scheme selection uses a two-stage check inside [`DeferredEstimate::Callback`]:
//! 1. Pass 1 (`expected_compression_ratio`) probes a zero-allocation stratified sample of valid
//!    elements and immediately skips general `f64`/`f32` columns in nanoseconds.
//! 2. Pass 2 (after every sampled float scheme has run and set `best_so_far`) confirms that 100%
//!    of full-array valid elements fit bit-exactly before estimating the narrowed child size, so
//!    an unsampled wider outlier can never leave the array uncompressed. It wins only when the
//!    narrowed child beats `best_so_far` by at least 2%.

use vortex_array::ArrayId;
use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::Canonical;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::aggregate_fn::fns::uncompressed_size_in_bytes::uncompressed_size_in_bytes;
use vortex_array::arrays::ChunkedArray;
use vortex_array::arrays::Constant;
use vortex_array::arrays::Masked;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::WidenArray;
use vortex_array::arrays::masked::MaskedArraySlotsExt;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::arrays::widen::encode::f32_slice_fits_in_f16;
use vortex_array::arrays::widen::encode::f64_slice_fits_in_f16;
use vortex_array::arrays::widen::encode::f64_slice_fits_in_f32;
use vortex_array::arrays::widen::encode::float_array_fits_in_ptype;
use vortex_array::arrays::widen::encode::narrow_float_values;
use vortex_array::arrays::widen::widen_v2_id;
use vortex_array::dtype::PType;
use vortex_compressor::sample_aligned_chunks;
use vortex_compressor::scheme::ChildSelection;
use vortex_compressor::scheme::CompressionEstimate;
use vortex_compressor::scheme::DeferredEstimate;
use vortex_compressor::scheme::DescendantExclusion;
use vortex_compressor::scheme::EstimateScore;
use vortex_compressor::scheme::EstimateVerdict;
use vortex_error::VortexExpect;
use vortex_error::VortexResult;
use vortex_mask::Mask;

use crate::ArrayAndStats;
use crate::CascadingCompressor;
use crate::CompressorContext;
use crate::Scheme;
use crate::SchemeExt;
use crate::schemes::float::FloatDictScheme;
use crate::schemes::float::FloatRLEScheme;

/// Number of elements per stratified probe window in Pass 1 and small-array sampling.
const SAMPLE_RUN_LEN: usize = 64;

/// Number of stratified runs sampled for Pass 1 probing and small-array ratio estimation.
const SAMPLE_RUNS: usize = 16;

/// Total rows covered by the small-array sample (`16 * 64 = 1024`, one FastLanes vector).
const SAMPLE_BUDGET: usize = SAMPLE_RUN_LEN * SAMPLE_RUNS;

/// FastLanes chunk size used when sampling long arrays as whole aligned chunks.
const ALIGNED_CHUNK_LEN: usize = 1024;

/// Minimum number of aligned chunks required before switching from 64-row runs to aligned chunks.
const MIN_ALIGNED_CHUNKS: usize = 8;

/// Minimum compression-ratio improvement over the best un-narrowed float scheme required to wrap
/// the array in a [`WidenArray`] (`>= 2%` smaller).
const MIN_IMPROVEMENT_FACTOR: f64 = 1.02;

/// Cached target width selected during the estimation callback so [`FloatWidenScheme::compress`]
/// avoids re-scanning the array.
#[derive(Debug, Clone, Copy)]
struct ChosenFloatWidenTarget(Option<PType>);

/// Lossless floating-point narrowing scheme (`f64 -> f32/f16`, `f32 -> f16`).
///
/// Produces `vortex.widen.v2` wire nodes, which belong to the `UNSTABLE` edition; builders whose
/// allowed encodings do not include `vortex.widen.v2` (or CUDA presets) drop this scheme.
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FloatWidenScheme;

impl Scheme for FloatWidenScheme {
    fn scheme_name(&self) -> &'static str {
        "vortex.float.widen"
    }

    fn matches(&self, canonical: &Canonical) -> bool {
        canonical.dtype().is_float()
    }

    fn produced_encodings(&self) -> Vec<ArrayId> {
        vec![widen_v2_id()]
    }

    /// Children: narrowed values = 0.
    fn num_children(&self) -> usize {
        1
    }

    /// Keep `Dict` and `RLE` at the root rather than under `Widen`.
    ///
    /// Leaving `Dict` at the root preserves DuckDB exporter and dict-layout fast paths; `Dict`
    /// and `RLE` still compress their values children through `FloatWidenScheme` when beneficial.
    fn descendant_exclusions(&self) -> Vec<DescendantExclusion> {
        vec![
            DescendantExclusion {
                excluded: FloatDictScheme.id(),
                children: ChildSelection::All,
            },
            DescendantExclusion {
                excluded: FloatRLEScheme.id(),
                children: ChildSelection::All,
            },
        ]
    }

    fn expected_compression_ratio(
        &self,
        data: &ArrayAndStats,
        _compress_ctx: CompressorContext,
        exec_ctx: &mut ExecutionCtx,
    ) -> CompressionEstimate {
        let primitive = data.array_as_primitive();
        let ptype = primitive.ptype();
        if ptype == PType::F16 || primitive.is_empty() {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        let Ok(mask) = primitive
            .validity()
            .and_then(|v| v.execute_mask(primitive.len(), exec_ctx))
        else {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        };
        if matches!(mask, Mask::AllFalse(_)) {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        // Fast zero-allocation Pass 1 probe exits in nanoseconds on general f64/f32 columns.
        if !probe_fits_any_narrower(primitive, &mask) {
            return CompressionEstimate::Verdict(EstimateVerdict::Skip);
        }

        CompressionEstimate::Deferred(DeferredEstimate::Callback(Box::new(
            move |compressor, data, best_so_far, compress_ctx, exec_ctx| {
                estimate_narrowed_ratio(compressor, data, best_so_far, &compress_ctx, exec_ctx)
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
        let primitive = data.array_as_primitive().into_owned();
        let mask = primitive
            .validity()?
            .execute_mask(primitive.len(), exec_ctx)?;
        let target = data
            .get_or_insert_with(|| ChosenFloatWidenTarget(narrowest_fit_ptype(&primitive, &mask)))
            .0;
        let Some(target) = target else {
            return Ok(data.array().clone());
        };

        let narrowed = narrow_float_values(&primitive, target, &mask)?;
        let compressed_child = compressor.compress_child(
            &narrowed.into_array(),
            &compress_ctx,
            self.id(),
            0,
            exec_ctx,
        )?;

        Ok(WidenArray::try_new(compressed_child, primitive.dtype().clone())?.into_array())
    }
}

/// Pass 2 callback: verifies full-array bit-exact fit, estimates compressed child size across
/// candidate narrower widths, and enforces the `>= 2%` savings guard over `best_so_far`.
fn estimate_narrowed_ratio(
    compressor: &CascadingCompressor,
    data: &ArrayAndStats,
    best_so_far: Option<EstimateScore>,
    compress_ctx: &CompressorContext,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<EstimateVerdict> {
    let primitive = data.array_as_primitive().into_owned();
    let ptype = primitive.ptype();
    let mask = primitive
        .validity()?
        .execute_mask(primitive.len(), exec_ctx)?;

    let candidates = candidate_targets_full(&primitive, &mask)?;
    if candidates.is_empty() {
        return Ok(EstimateVerdict::Skip);
    }

    let threshold = best_so_far.and_then(EstimateScore::finite_ratio);

    // When the cascade budget is exhausted, `compress_child` keeps the narrowed primitive values
    // uncompressed (with compressed validity), so the narrowest target always minimizes size.
    if compress_ctx.finished_cascading() {
        let best_target = candidates[0];
        let before = uncompressed_size_in_bytes(data.array(), exec_ctx)? as u64;
        let validity_bytes = before
            .saturating_sub(u64::try_from(primitive.len() * ptype.byte_width()).unwrap_or(before));
        let after = u64::try_from(primitive.len() * best_target.byte_width())
            .unwrap_or(before)
            .saturating_add(validity_bytes);
        if after == 0 {
            return Ok(EstimateVerdict::Skip);
        }
        let ratio = before as f64 / after as f64;
        if ratio < threshold.unwrap_or(1.0) * MIN_IMPROVEMENT_FACTOR {
            return Ok(EstimateVerdict::Skip);
        }
        data.get_or_insert_with(|| ChosenFloatWidenTarget(Some(best_target)));
        return Ok(EstimateVerdict::Ratio(ratio));
    }

    let (sample_prim, is_subsampled) =
        build_sample_primitive(data.array(), compress_ctx.is_sample(), exec_ctx)?;
    let sample_mask = sample_prim
        .validity()?
        .execute_mask(sample_prim.len(), exec_ctx)?;
    if matches!(sample_mask, Mask::AllFalse(_)) {
        return Ok(EstimateVerdict::Skip);
    }
    let sample_before = uncompressed_size_in_bytes(sample_prim.as_ref(), exec_ctx)? as u64;
    if sample_before == 0 {
        return Ok(EstimateVerdict::Skip);
    }

    let mut best: Option<(PType, u64)> = None;
    for target in candidates {
        let narrowed_sample = narrow_float_values(&sample_prim, target, &sample_mask)?;
        let compressed_sample = compressor.compress_child(
            &narrowed_sample.into_array(),
            compress_ctx,
            FloatWidenScheme.id(),
            0,
            exec_ctx,
        )?;
        // A subsampled slice that happens to be constant can trigger root constant folding even
        // though the full array is non-constant; reject that degenerate ratio estimate.
        if is_subsampled && is_constant_or_masked_constant(&compressed_sample) {
            continue;
        }
        let sample_after = compressed_sample.nbytes();
        if sample_after == 0 {
            continue;
        }
        if best.is_none_or(|(_, best_after)| sample_after < best_after) {
            best = Some((target, sample_after));
        }
    }

    let Some((best_target, best_after)) = best else {
        return Ok(EstimateVerdict::Skip);
    };
    let estimated_ratio = sample_before as f64 / best_after as f64;
    if estimated_ratio < threshold.unwrap_or(1.0) * MIN_IMPROVEMENT_FACTOR {
        return Ok(EstimateVerdict::Skip);
    }

    data.get_or_insert_with(|| ChosenFloatWidenTarget(Some(best_target)));
    Ok(EstimateVerdict::Ratio(estimated_ratio))
}

fn is_constant_or_masked_constant(array: &ArrayRef) -> bool {
    array.is::<Constant>()
        || array
            .as_opt::<Masked>()
            .is_some_and(|masked| masked.child().is::<Constant>())
}

/// Fast zero-allocation Pass 1 probe checking whether sampled valid elements fit in the widest
/// strictly narrower float type (`f32` for `f64`, `f16` for `f32`).
fn probe_fits_any_narrower(array: ArrayView<'_, Primitive>, mask: &Mask) -> bool {
    match array.ptype() {
        PType::F64 => probe_valid_slice(array.as_slice::<f64>(), mask, f64_slice_fits_in_f32),
        PType::F32 => probe_valid_slice(array.as_slice::<f32>(), mask, f32_slice_fits_in_f16),
        _ => false,
    }
}

fn probe_valid_slice<T: Copy>(slice: &[T], mask: &Mask, fits_slice: impl Fn(&[T]) -> bool) -> bool {
    match mask {
        Mask::AllTrue(_) => {
            let len = slice.len();
            if len <= SAMPLE_BUDGET {
                return fits_slice(slice);
            }
            let step = (len - SAMPLE_RUN_LEN) / (SAMPLE_RUNS - 1);
            for run in 0..SAMPLE_RUNS {
                let start = run * step;
                if !fits_slice(&slice[start..start + SAMPLE_RUN_LEN]) {
                    return false;
                }
            }
            true
        }
        Mask::AllFalse(_) => false,
        Mask::Values(values) => {
            let slices = values.slices();
            if slices.len() <= SAMPLE_RUNS {
                for &(start, end) in slices {
                    let run_len = end - start;
                    if run_len <= SAMPLE_RUN_LEN {
                        if !fits_slice(&slice[start..end]) {
                            return false;
                        }
                    } else if !fits_slice(&slice[start..start + SAMPLE_RUN_LEN])
                        || !fits_slice(&slice[end - SAMPLE_RUN_LEN..end])
                    {
                        return false;
                    }
                }
                true
            } else {
                let step = (slices.len() - 1) / (SAMPLE_RUNS - 1);
                for run in 0..SAMPLE_RUNS {
                    let (start, end) = slices[run * step];
                    let probe_end = (start + SAMPLE_RUN_LEN).min(end);
                    if !fits_slice(&slice[start..probe_end]) {
                        return false;
                    }
                }
                true
            }
        }
    }
}

/// Verifies full-array bit-exact fit and returns all valid narrower targets ordered narrowest
/// first (`[F16, F32]` when an `f64` array fits both, because `ALP`/`ALPRD` skip `f16` so `f32`
/// can still beat `f16` after child compression).
fn candidate_targets_full(array: &PrimitiveArray, mask: &Mask) -> VortexResult<Vec<PType>> {
    match array.ptype() {
        PType::F64 => {
            if !float_array_fits_in_ptype(array, PType::F32, mask)? {
                return Ok(Vec::new());
            }
            // Probe `f16` before scanning the full array so `f32`-only columns exit immediately.
            let fits_f16 = probe_valid_slice(array.as_slice::<f64>(), mask, f64_slice_fits_in_f16)
                && float_array_fits_in_ptype(array, PType::F16, mask)?;
            if fits_f16 {
                Ok(vec![PType::F16, PType::F32])
            } else {
                Ok(vec![PType::F32])
            }
        }
        PType::F32 => {
            if float_array_fits_in_ptype(array, PType::F16, mask)? {
                Ok(vec![PType::F16])
            } else {
                Ok(Vec::new())
            }
        }
        _ => Ok(Vec::new()),
    }
}

/// Fallback full-array scan used only if `FloatWidenScheme::compress` is invoked directly without
/// running its estimation callback first.
fn narrowest_fit_ptype(array: &PrimitiveArray, mask: &Mask) -> Option<PType> {
    candidate_targets_full(array, mask)
        .ok()
        .and_then(|targets| targets.into_iter().next())
}

/// Builds a deterministic ~1% sample executed to a contiguous [`PrimitiveArray`], returning
/// `(sample, is_subsampled)`.
fn build_sample_primitive(
    array: &ArrayRef,
    already_sample: bool,
    exec_ctx: &mut ExecutionCtx,
) -> VortexResult<(PrimitiveArray, bool)> {
    let len = array.len();
    if already_sample || len <= SAMPLE_BUDGET {
        let prim = array.clone().execute::<PrimitiveArray>(exec_ctx)?;
        return Ok((prim, false));
    }

    let approx_one_pct_chunks = (len / 100) / ALIGNED_CHUNK_LEN;
    if approx_one_pct_chunks >= MIN_ALIGNED_CHUNKS {
        let chunk_count =
            u32::try_from(approx_one_pct_chunks).vortex_expect("chunk count fits in u32");
        if let Some(sampled) = sample_aligned_chunks(array, ALIGNED_CHUNK_LEN, chunk_count) {
            return Ok((sampled.execute::<PrimitiveArray>(exec_ctx)?, true));
        }
    }

    let step = (len - SAMPLE_RUN_LEN) / (SAMPLE_RUNS - 1);
    let runs = (0..SAMPLE_RUNS)
        .map(|run| {
            let start = run * step;
            array.slice(start..start + SAMPLE_RUN_LEN)
        })
        .collect::<VortexResult<Vec<_>>>()?;
    let sampled = ChunkedArray::try_new(runs, array.dtype().clone())?.into_array();
    Ok((sampled.execute::<PrimitiveArray>(exec_ctx)?, true))
}

#[cfg(test)]
mod tests {
    use std::sync::LazyLock;

    use vortex_array::ArrayContext;
    use vortex_array::ArrayRef;
    use vortex_array::ExecutionCtx;
    use vortex_array::IntoArray;
    use vortex_array::VortexSessionExecute;
    use vortex_array::arrays::PrimitiveArray;
    use vortex_array::arrays::Widen;
    use vortex_array::arrays::widen::WidenArraySlotsExt;
    use vortex_array::arrays::widen::widen_v2_id;
    use vortex_array::assert_arrays_eq;
    use vortex_array::dtype::PType;
    use vortex_array::dtype::half::f16;
    use vortex_array::serde::SerializeOptions;
    use vortex_array::serde::SerializedArray;
    use vortex_array::validity::Validity;
    use vortex_buffer::Buffer;
    use vortex_buffer::ByteBufferMut;
    use vortex_error::VortexResult;
    use vortex_session::VortexSession;
    use vortex_session::registry::ReadContext;

    use super::FloatWidenScheme;
    use crate::BtrBlocksCompressor;
    use crate::BtrBlocksCompressorBuilder;
    use crate::CompressionSession;
    use crate::SchemeExt;

    static SESSION: LazyLock<VortexSession> = LazyLock::new(|| {
        let session = VortexSession::empty().with::<CompressionSession>();
        vortex_array::initialize(&session);
        session
    });

    fn unrestricted_compressor() -> BtrBlocksCompressor {
        BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .build()
    }

    /// Deterministic pseudorandom f32-exact values widened to f64 (modeled after OpenAI embeddings
    /// stored as f64) so ALP cannot exploit small decimal scale and ALPRD+Widen wins decisively.
    fn pseudo_random_f32_in_f64(len: usize) -> Vec<f64> {
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        (0..len)
            .map(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let u = ((state >> 40) as f32) / ((1u32 << 24) as f32) - 0.5_f32;
                f64::from(u)
            })
            .collect()
    }

    fn assert_bit_exact_f64(
        actual: &ArrayRef,
        expected: &PrimitiveArray,
        ctx: &mut ExecutionCtx,
    ) -> VortexResult<()> {
        let actual_prim = actual.clone().execute::<PrimitiveArray>(ctx)?;
        assert_arrays_eq!(actual_prim, expected, ctx);
        let mask = expected.validity()?.execute_mask(expected.len(), ctx)?;
        let actual_slice = actual_prim.as_slice::<f64>();
        let expected_slice = expected.as_slice::<f64>();
        for i in 0..expected.len() {
            if mask.value(i) {
                assert_eq!(
                    actual_slice[i].to_bits(),
                    expected_slice[i].to_bits(),
                    "bit mismatch at row {i}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn compresses_f32_exact_f64_with_widen_and_saves_half() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let values = pseudo_random_f32_in_f64(8192);
        let input = PrimitiveArray::from_iter(values);

        let without_widen = BtrBlocksCompressorBuilder::from_session(&SESSION)
            .unrestricted()
            .exclude_schemes([FloatWidenScheme.id()])
            .build()
            .compress(&input.clone().into_array(), &mut ctx)?;
        let with_widen =
            unrestricted_compressor().compress(&input.clone().into_array(), &mut ctx)?;

        assert!(
            with_widen.is::<Widen>(),
            "expected Widen at root, got {}",
            with_widen.encoding_id()
        );
        assert_eq!(
            with_widen.as_::<Widen>().values().dtype().as_ptype(),
            PType::F32
        );
        assert!(
            with_widen.nbytes() * 100 < without_widen.nbytes() * 60,
            "expected >40% savings, got {} vs {}",
            with_widen.nbytes(),
            without_widen.nbytes()
        );
        assert_bit_exact_f64(&with_widen, &input, &mut ctx)?;

        Ok(())
    }

    #[test]
    fn preserves_neg_zero_and_canonical_nan_bits() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut values = pseudo_random_f32_in_f64(4096);
        values[0] = -0.0_f64;
        values[1] = 0.0_f64;
        values[2] = f64::INFINITY;
        values[3] = f64::NEG_INFINITY;
        values[4] = f64::NAN;
        values[5] = -f64::NAN;
        let input = PrimitiveArray::from_iter(values);

        let compressed =
            unrestricted_compressor().compress(&input.clone().into_array(), &mut ctx)?;
        assert!(compressed.is::<Widen>());
        assert_bit_exact_f64(&compressed, &input, &mut ctx)?;

        Ok(())
    }

    #[test]
    fn skips_unsampled_non_f32_outlier_and_still_compresses_normally() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut values = pseudo_random_f32_in_f64(16_384);
        // Place a single non-f32-exact value outside the 16x64 probe windows.
        values[100] = 0.1_f64;
        let input = PrimitiveArray::from_iter(values);

        let compressed =
            unrestricted_compressor().compress(&input.clone().into_array(), &mut ctx)?;
        assert!(!compressed.is::<Widen>());
        // Must still compress with ALPRD rather than bailing out to uncompressed Primitive.
        assert!(
            compressed.nbytes() < input.nbytes(),
            "expected fallback float scheme to compress, got {} vs {}",
            compressed.nbytes(),
            input.nbytes()
        );
        assert_bit_exact_f64(&compressed, &input, &mut ctx)?;

        Ok(())
    }

    #[test]
    fn skips_signaling_nan() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut values = pseudo_random_f32_in_f64(4096);
        values[77] = f64::from_bits(0x7ff0_0000_2000_0000);
        let input = PrimitiveArray::from_iter(values);

        let compressed =
            unrestricted_compressor().compress(&input.clone().into_array(), &mut ctx)?;
        assert!(!compressed.is::<Widen>());
        assert_bit_exact_f64(&compressed, &input, &mut ctx)?;

        Ok(())
    }

    #[test]
    fn ignores_non_f32_null_garbage() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut values = pseudo_random_f32_in_f64(4096);
        let mut valid = vec![true; values.len()];
        for i in (0..values.len()).step_by(13) {
            values[i] = 0.1_f64;
            valid[i] = false;
        }
        let input = PrimitiveArray::new(Buffer::from(values), Validity::from_iter(valid));

        let compressed =
            unrestricted_compressor().compress(&input.clone().into_array(), &mut ctx)?;
        assert!(compressed.is::<Widen>());
        assert_bit_exact_f64(&compressed, &input, &mut ctx)?;

        Ok(())
    }

    #[test]
    fn compresses_f16_exact_f32_to_f16_widen() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let mut state = 0x1234_5678u32;
        let values: Vec<f32> = (0..4096)
            .map(|_| {
                state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                // Generate diverse finite f16 bit patterns (exponent 1..28, full mantissa).
                let exp = u16::try_from((state >> 24) % 28 + 1).unwrap_or(1);
                let mant = (state & 0x03ff) as u16;
                let sign = ((state >> 31) as u16) << 15;
                f16::from_bits(sign | (exp << 10) | mant).to_f32()
            })
            .collect();
        let input = PrimitiveArray::from_iter(values);

        let compressed =
            unrestricted_compressor().compress(&input.clone().into_array(), &mut ctx)?;
        assert!(compressed.is::<Widen>());
        assert_eq!(
            compressed.as_::<Widen>().values().dtype().as_ptype(),
            PType::F16
        );
        assert_arrays_eq!(compressed, input, &mut ctx);

        Ok(())
    }

    #[test]
    fn wire_round_trip_uses_widen_v2_id() -> VortexResult<()> {
        let mut ctx = SESSION.create_execution_ctx();
        let input = PrimitiveArray::from_iter(pseudo_random_f32_in_f64(4096));
        let compressed = BtrBlocksCompressorBuilder::empty()
            .with_new_scheme(&FloatWidenScheme)
            .build()
            .compress(&input.clone().into_array(), &mut ctx)?;
        assert!(compressed.is::<Widen>());

        let array_ctx = ArrayContext::empty();
        let mut blob = ByteBufferMut::empty();
        for buf in compressed.serialize(&array_ctx, &SESSION, &SerializeOptions::default())? {
            blob.extend_from_slice(buf.as_ref());
        }
        let ids = array_ctx.to_ids();
        assert!(ids.contains(&widen_v2_id()), "expected widen.v2 in {ids:?}");

        let serialized = SerializedArray::try_from(blob.freeze())?;
        let decoded = serialized.decode(
            compressed.dtype(),
            compressed.len(),
            &ReadContext::new(ids),
            &SESSION,
        )?;
        assert_bit_exact_f64(&decoded, &input, &mut ctx)?;

        Ok(())
    }
}
