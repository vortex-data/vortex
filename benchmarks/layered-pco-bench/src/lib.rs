// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Composition harness for measuring a "pco-style top + btrblocks-bottom"
//! hybrid compressor against full pco and vanilla btrblocks.
//!
//! See the binary in `main.rs` for the analysis script that uses this
//! library. The library exposes [`hybrid_compress`] and a small set of
//! helpers so the bench binary stays focused on data generation, timing,
//! and reporting.
//!
//! The hybrid is intentionally *not* a new Vortex array. It is a compose-
//! and-measure script that builds an array tree by hand from the existing
//! layered-pco P1–P3 arrays and then compresses each leaf with
//! [`BtrBlocksCompressor`]. This sidesteps the need for a new VTable while
//! still exercising the architecture we want to evaluate: pco's structural
//! decorrelations on top, a cascading scheme picker on the bottom.

use vortex_array::ArrayRef;
use vortex_array::ArrayView;
use vortex_array::ExecutionCtx;
use vortex_array::IntoArray;
use vortex_array::arrays::Primitive;
use vortex_array::arrays::PrimitiveArray;
use vortex_array::arrays::primitive::PrimitiveArrayExt;
use vortex_array::dtype::PType;
use vortex_btrblocks::BtrBlocksCompressor;
use vortex_consecutive_delta::ConsecutiveDelta;
use vortex_consecutive_delta::ConsecutiveDeltaArrayExt;
use vortex_error::VortexResult;
use vortex_error::vortex_bail;
use vortex_ordered_latent::OrderedLatent;
use vortex_ordered_latent::OrderedLatentArrayExt;

/// Skip btrblocks on small leaves — its scheme-selection overhead exceeds
/// any savings on tiny arrays. Tuned by inspection; not a tight bound.
const SMALL_LEAF_THRESHOLD_BYTES: u64 = 1024;

/// Sample size used by the heuristics to inspect the input cheaply.
const SAMPLE_SIZE: usize = 1024;

/// One row of a sampled trace; the heuristics consume a slice of these.
#[derive(Copy, Clone)]
struct DeltaStats {
    /// Variance of the raw value sample.
    var_value: f64,
    /// Variance of the first-order differences of the sample.
    var_delta: f64,
    /// Number of differences observed in the sample.
    n_deltas: usize,
}

/// Compress a primitive array via the hybrid:
///
/// 1. (Optional) `ConsecutiveDelta` when the input is monotone-ish i64.
/// 2. Otherwise `OrderedLatent` to push values into an unsigned latent.
/// 3. Compress every primitive leaf produced by steps 1-2 with
///    [`BtrBlocksCompressor`], skipping leaves smaller than
///    [`SMALL_LEAF_THRESHOLD_BYTES`].
///
/// This deliberately leaves Mode (IntMult / FloatMult / FloatQuant / Dict)
/// off the auto-picked path. The two evaluation datasets do not benefit
/// from any of those modes; adding them would not change the measurement
/// and would clutter the heuristic with un-exercised branches.
pub fn hybrid_compress(
    parray: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ptype = PrimitiveArrayExt::ptype(&parray);
    let compressor = BtrBlocksCompressor::default();

    if ptype == PType::I64 && looks_monotone_i64(&parray)? {
        encode_consec_delta_then_btrblocks(parray, &compressor, ctx)
    } else {
        encode_ordered_latent_then_btrblocks(parray, &compressor, ctx)
    }
}

fn encode_consec_delta_then_btrblocks(
    parray: ArrayView<'_, Primitive>,
    compressor: &BtrBlocksCompressor,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let cd = ConsecutiveDelta::encode(parray, ctx)?;
    let primary = cd.primary().clone();
    let compressed_primary = compress_leaf(&primary, compressor, ctx)?;
    cd.into_array().with_slot(0, compressed_primary)
}

fn encode_ordered_latent_then_btrblocks(
    parray: ArrayView<'_, Primitive>,
    compressor: &BtrBlocksCompressor,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ol = OrderedLatent::encode(parray, ctx)?;
    let encoded = ol.encoded().clone();
    let compressed_encoded = compress_leaf(&encoded, compressor, ctx)?;
    ol.into_array().with_slot(0, compressed_encoded)
}

/// Compress a single leaf, skipping leaves that are smaller than
/// [`SMALL_LEAF_THRESHOLD_BYTES`]. Skipping is by raw `nbytes()`, which
/// already counts only the leaf's own buffers.
fn compress_leaf(
    leaf: &ArrayRef,
    compressor: &BtrBlocksCompressor,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    if leaf.nbytes() < SMALL_LEAF_THRESHOLD_BYTES {
        return Ok(leaf.clone());
    }
    compressor.compress(leaf, ctx)
}

/// Heuristic: sample up to `SAMPLE_SIZE` evenly-spaced positions and
/// compare the variance of consecutive differences to the variance of raw
/// values. When the delta variance is at least an order of magnitude
/// smaller, treat the column as monotone-ish.
fn looks_monotone_i64(parray: &ArrayView<'_, Primitive>) -> VortexResult<bool> {
    let len = parray.array().len();
    if len < 2 {
        return Ok(false);
    }
    let buf = parray.as_slice::<i64>();
    let stats = sample_delta_stats_i64(buf);
    if stats.n_deltas == 0 || stats.var_value == 0.0 {
        return Ok(false);
    }
    // The two evaluation datasets sit at extreme ends of this ratio:
    // monotone-with-jitter has var_delta / var_value ≈ 1e-13, and the
    // cube-distributed scenario has it ≈ 1 (deltas roughly as noisy as
    // values). A 100x gap is a generous separator.
    Ok(stats.var_delta * 100.0 < stats.var_value)
}

fn sample_delta_stats_i64(values: &[i64]) -> DeltaStats {
    let len = values.len();
    let stride = len.div_ceil(SAMPLE_SIZE).max(1);
    let mut sampled: Vec<i64> = (0..len).step_by(stride).map(|i| values[i]).collect();
    if sampled.len() < 2 {
        sampled = values.iter().copied().take(2).collect();
    }
    let var_value = sample_variance_i64(&sampled);
    let deltas: Vec<f64> = sampled
        .windows(2)
        .map(|w| w[1].wrapping_sub(w[0]) as f64)
        .collect();
    let var_delta = sample_variance_f64(&deltas);
    DeltaStats {
        var_value,
        var_delta,
        n_deltas: deltas.len(),
    }
}

fn sample_variance_i64(values: &[i64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().map(|&v| v as f64).sum::<f64>() / n;
    values
        .iter()
        .map(|&v| {
            let d = v as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / n
}

fn sample_variance_f64(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    values.iter().map(|&v| (v - mean).powi(2)).sum::<f64>() / n
}

/// Compose the "plain layered" baseline: pco-style structural top, but
/// with raw `PrimitiveArray` leaves (no entropy coder, no btrblocks).
///
/// This is the bottom-of-the-curve point: how much of pco's compression
/// is already accomplished by its decorrelations alone, before any
/// entropy code at all.
pub fn layered_plain_compress(
    parray: ArrayView<'_, Primitive>,
    ctx: &mut ExecutionCtx,
) -> VortexResult<ArrayRef> {
    let ptype = PrimitiveArrayExt::ptype(&parray);
    if ptype == PType::I64 && looks_monotone_i64(&parray)? {
        let cd = ConsecutiveDelta::encode(parray, ctx)?;
        Ok(cd.into_array())
    } else {
        let ol = OrderedLatent::encode(parray, ctx)?;
        Ok(ol.into_array())
    }
}

/// Round-trip sanity check: decode `compressed` back to a canonical
/// `PrimitiveArray` and verify it equals `expected`.
///
/// Used from the analysis binary; lives here so it shares the
/// `VortexSessionExecute` import discipline.
pub fn check_roundtrip(
    compressed: ArrayRef,
    expected: &PrimitiveArray,
    ctx: &mut ExecutionCtx,
) -> VortexResult<()> {
    let decoded = compressed.execute::<PrimitiveArray>(ctx)?;
    if decoded.len() != expected.len() {
        vortex_bail!(
            "round-trip length mismatch: expected {}, got {}",
            expected.len(),
            decoded.len()
        );
    }
    let decoded_slice = decoded.as_slice::<i64>();
    let expected_slice = expected.as_slice::<i64>();
    if decoded_slice != expected_slice {
        let first_diff = decoded_slice
            .iter()
            .zip(expected_slice)
            .position(|(a, b)| a != b)
            .map(|i| (i, decoded_slice[i], expected_slice[i]));
        vortex_bail!("round-trip value mismatch at first diff: {:?}", first_diff);
    }
    Ok(())
}
