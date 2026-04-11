// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Correctness verification for vector-search variants.
//!
//! Before the timing loop runs, we compute cosine-similarity scores for a single query
//! row against the uncompressed baseline and against each prepared variant, then compare
//! the two score vectors element-by-element. This catches two distinct classes of bug:
//!
//! - A **lossless variant** that disagrees with the uncompressed scan (bug in the
//!   compression pipeline, or in how we're routing through the scalar-fn dispatch, or in
//!   the variant-specific decompress path).
//! - A **lossy variant** (TurboQuant) that drifts further from ground truth than we
//!   expect from the bit-width and SORF rotation settings (regression in the encoder).
//!
//! The same `execute_cosine` function the timing loop uses is also what verification
//! uses, so the correctness check is validating the *exact* expression tree we're about
//! to benchmark. Lossless variants must match within [`LOSSLESS_TOLERANCE`]; lossy
//! variants must match within [`LOSSY_TOLERANCE`]. A hard-stop `Err` return on any
//! mismatch keeps the benchmark honest — you cannot publish throughput numbers for a
//! variant that's returning garbage.

use anyhow::Result;
use anyhow::bail;
use vortex::array::ArrayRef;
use vortex::array::VortexSessionExecute;
use vortex::session::VortexSession;

use crate::execute_cosine;

/// Maximum acceptable absolute difference in cosine scores for a *lossless* variant
/// (uncompressed, BtrBlocks-default). `cosine_similarity` traverses the FSL storage and
/// reduces with f32 accumulators, so a pure algebraic change of encoding can shift a
/// score by a few ULPs of f32 precision. `1e-4` is well above that noise floor while
/// still catching real regressions.
pub const LOSSLESS_TOLERANCE: f32 = 1e-4;

/// Maximum acceptable absolute difference in cosine scores for the *lossy* TurboQuant
/// variant. At the default 8-bit configuration the reconstructed dot product typically
/// drifts by well under 0.05 for unit-normalized vectors. `0.2` is a loose upper bound
/// that catches regressions without flaking on distribution-specific noise.
pub const LOSSY_TOLERANCE: f32 = 0.2;

/// How lossy a variant is allowed to be when its scores are compared to the
/// uncompressed baseline.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerificationKind {
    /// Lossless variants must match within [`LOSSLESS_TOLERANCE`].
    Lossless,
    /// Lossy variants must match within [`LOSSY_TOLERANCE`].
    Lossy,
}

/// Per-variant correctness report. Captured for both pass and fail outcomes so the
/// caller can emit the numbers as dashboard measurements regardless.
#[derive(Debug, Clone, Copy)]
pub struct VerificationReport {
    /// Number of rows compared (== dataset row count).
    pub num_scores: usize,
    /// Mean absolute difference between baseline and variant cosine scores.
    pub mean_abs_diff: f64,
    /// Max absolute difference between baseline and variant cosine scores.
    pub max_abs_diff: f64,
    /// Which tolerance band applied.
    pub kind: VerificationKind,
    /// Whether the variant's max-abs-diff stayed within its tolerance.
    pub passed: bool,
}

impl VerificationReport {
    /// The tolerance that was applied to produce [`Self::passed`].
    pub fn tolerance(&self) -> f32 {
        match self.kind {
            VerificationKind::Lossless => LOSSLESS_TOLERANCE,
            VerificationKind::Lossy => LOSSY_TOLERANCE,
        }
    }
}

/// Compute cosine-similarity scores for a single query row on `data` and return them
/// as a plain `Vec<f32>`. This is just a convenience wrapper around
/// [`crate::execute_cosine`] that pulls the f32 slice out of the resulting
/// `PrimitiveArray`.
pub fn compute_cosine_scores(
    data: &ArrayRef,
    query: &[f32],
    session: &VortexSession,
) -> Result<Vec<f32>> {
    let mut ctx = session.create_execution_ctx();
    let scores = execute_cosine(data, query, &mut ctx)?;
    Ok(scores.as_slice::<f32>().to_vec())
}

/// Compare two equal-length score vectors and return their mean absolute difference
/// and max absolute difference, without evaluating a pass/fail threshold.
pub fn compare_scores(baseline: &[f32], other: &[f32]) -> (f64, f64) {
    assert_eq!(
        baseline.len(),
        other.len(),
        "compare_scores: length mismatch baseline={} other={}",
        baseline.len(),
        other.len(),
    );

    if baseline.is_empty() {
        return (0.0, 0.0);
    }

    let mut sum = 0.0f64;
    let mut max: f64 = 0.0;
    for (&b, &o) in baseline.iter().zip(other.iter()) {
        // Treat (+0, -0) pairs as equal and propagate NaN as inf so it always fails
        // the tolerance check below.
        let diff = if b.is_nan() || o.is_nan() {
            f64::INFINITY
        } else {
            (f64::from(b) - f64::from(o)).abs()
        };
        sum += diff;
        if diff > max {
            max = diff;
        }
    }
    (sum / baseline.len() as f64, max)
}

/// Verify one variant's scores against a baseline and produce a full
/// [`VerificationReport`]. Whether `passed` is true depends on `kind`'s tolerance.
pub fn verify_scores(
    baseline: &[f32],
    variant_scores: &[f32],
    kind: VerificationKind,
) -> VerificationReport {
    let (mean_abs_diff, max_abs_diff) = compare_scores(baseline, variant_scores);
    let tolerance = match kind {
        VerificationKind::Lossless => f64::from(LOSSLESS_TOLERANCE),
        VerificationKind::Lossy => f64::from(LOSSY_TOLERANCE),
    };
    let passed = max_abs_diff <= tolerance;
    VerificationReport {
        num_scores: baseline.len(),
        mean_abs_diff,
        max_abs_diff,
        kind,
        passed,
    }
}

/// Verify pre-computed scores against a baseline and enforce the tolerance band.
///
/// Takes already-materialized `variant_scores` (as a `&[f32]`) rather than an
/// `ArrayRef`, so both the Vortex-variant path (which computes scores via
/// [`execute_cosine`](crate::execute_cosine)) and the hand-rolled baseline path (which
/// runs a plain Rust loop over a flat `Vec<f32>`) share the same error-handling,
/// logging, and hard-fail logic without duplicating it in `main.rs`.
///
/// Lossless mismatches bail the run with an error; lossy mismatches log a warning
/// but let the run continue so the recall measurement is still reported.
pub fn verify_and_report_scores(
    variant_name: &str,
    variant_scores: &[f32],
    baseline_scores: &[f32],
    kind: VerificationKind,
) -> Result<VerificationReport> {
    let report = verify_scores(baseline_scores, variant_scores, kind);

    if !report.passed {
        let message = format!(
            "{variant_name} correctness check failed: max_abs_diff={:.6}, \
             mean_abs_diff={:.6}, tolerance={:.6} ({:?})",
            report.max_abs_diff,
            report.mean_abs_diff,
            report.tolerance(),
            report.kind,
        );
        match kind {
            VerificationKind::Lossless => bail!("{message}"),
            VerificationKind::Lossy => {
                tracing::warn!("{message}");
            }
        }
    }

    Ok(report)
}

/// End-to-end variant verification: executes cosine on `variant_array` against the
/// same query used for the baseline and returns a [`VerificationReport`]. Returns
/// `Err` if `kind` is [`VerificationKind::Lossless`] and the scores disagree beyond
/// [`LOSSLESS_TOLERANCE`] — that indicates a real correctness bug, not a quality
/// tradeoff.
pub fn verify_variant(
    variant_name: &str,
    variant_array: &ArrayRef,
    query: &[f32],
    baseline_scores: &[f32],
    kind: VerificationKind,
    session: &VortexSession,
) -> Result<VerificationReport> {
    let scores = compute_cosine_scores(variant_array, query, session)?;
    verify_and_report_scores(variant_name, &scores, baseline_scores, kind)
}

#[cfg(test)]
mod tests {
    use vortex_bench::SESSION;

    use super::*;
    use crate::Variant;
    use crate::prepare_variant;
    use crate::test_utils::synthetic_vector;

    fn make_prepared(dim: u32, num_rows: usize, seed: u64) -> crate::PreparedDataset {
        let uncompressed = synthetic_vector(dim, num_rows, seed);
        crate::PreparedDataset {
            name: "synthetic".to_string(),
            uncompressed,
            // Filled in below from row 0.
            query: vec![],
            parquet_bytes: 0,
        }
    }

    fn extract_row_zero(uncompressed: &ArrayRef, dim: u32) -> Vec<f32> {
        use vortex::array::VortexSessionExecute;
        use vortex::array::arrays::Extension;
        use vortex::array::arrays::FixedSizeListArray;
        use vortex::array::arrays::PrimitiveArray;
        use vortex::array::arrays::extension::ExtensionArrayExt;
        use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;

        let mut ctx = SESSION.create_execution_ctx();
        let ext = uncompressed.as_opt::<Extension>().unwrap();
        let fsl: FixedSizeListArray = ext.storage_array().clone().execute(&mut ctx).unwrap();
        let elements: PrimitiveArray = fsl.elements().clone().execute(&mut ctx).unwrap();
        elements.as_slice::<f32>()[..dim as usize].to_vec()
    }

    #[test]
    fn compare_scores_handles_empty() {
        let (mean, max) = compare_scores(&[], &[]);
        assert_eq!(mean, 0.0);
        assert_eq!(max, 0.0);
    }

    #[test]
    fn compare_scores_computes_mae_and_max() {
        let base = [0.0f32, 1.0, 2.0, 3.0];
        let other = [0.0f32, 1.0, 2.5, 3.0];
        let (mean, max) = compare_scores(&base, &other);
        assert!((max - 0.5).abs() < 1e-9);
        assert!((mean - 0.125).abs() < 1e-9);
    }

    #[test]
    fn verify_scores_passes_for_identical_inputs() {
        let base = [0.5f32; 10];
        let report = verify_scores(&base, &base, VerificationKind::Lossless);
        assert!(report.passed);
        assert_eq!(report.max_abs_diff, 0.0);
        assert_eq!(report.mean_abs_diff, 0.0);
        assert_eq!(report.num_scores, 10);
    }

    #[test]
    fn verify_scores_fails_for_lossless_beyond_tolerance() {
        let base = [0.5f32; 10];
        let mut other = [0.5f32; 10];
        other[3] = 0.50001; // diff ≈ 1e-5, comfortably below the 1e-4 lossless bound
        let report_ok = verify_scores(&base, &other, VerificationKind::Lossless);
        assert!(
            report_ok.passed,
            "1e-5 drift should pass, got max={:.2e}",
            report_ok.max_abs_diff
        );

        other[3] = 0.51; // diff of 0.01, well above 1e-4
        let report_bad = verify_scores(&base, &other, VerificationKind::Lossless);
        assert!(
            !report_bad.passed,
            "1e-2 drift should fail, got max={:.2e}",
            report_bad.max_abs_diff
        );
    }

    #[test]
    fn verify_scores_lossy_tolerates_small_drift() {
        let base = [0.9f32; 10];
        let mut other = [0.9f32; 10];
        other[0] = 1.0; // diff of 0.1
        let report = verify_scores(&base, &other, VerificationKind::Lossy);
        assert!(
            report.passed,
            "0.1 drift should pass lossy tolerance, got max={}",
            report.max_abs_diff
        );
    }

    #[test]
    fn verify_scores_fails_on_nan() {
        let base = [0.5f32, 0.5];
        let other = [0.5f32, f32::NAN];
        let report = verify_scores(&base, &other, VerificationKind::Lossless);
        assert!(!report.passed);
        assert!(report.max_abs_diff.is_infinite());
    }

    #[test]
    fn vortex_default_matches_uncompressed_end_to_end() {
        let dim = 128u32;
        let num_rows = 64usize;
        let mut prepared = make_prepared(dim, num_rows, 0xC0FFEE);
        prepared.query = extract_row_zero(&prepared.uncompressed, dim);

        let baseline_scores =
            compute_cosine_scores(&prepared.uncompressed, &prepared.query, &SESSION).unwrap();

        let default_prep = prepare_variant(&prepared, Variant::VortexDefault, &SESSION).unwrap();
        let report = verify_variant(
            "vortex-default",
            &default_prep.array,
            &prepared.query,
            &baseline_scores,
            VerificationKind::Lossless,
            &SESSION,
        )
        .expect("vortex-default must be lossless against the uncompressed baseline");
        assert!(report.passed);
    }

    #[test]
    fn vortex_turboquant_stays_within_lossy_tolerance() {
        let dim = 128u32;
        let num_rows = 64usize;
        let mut prepared = make_prepared(dim, num_rows, 0xDEADBEEF);
        prepared.query = extract_row_zero(&prepared.uncompressed, dim);

        let baseline_scores =
            compute_cosine_scores(&prepared.uncompressed, &prepared.query, &SESSION).unwrap();

        let tq_prep = prepare_variant(&prepared, Variant::VortexTurboQuant, &SESSION).unwrap();
        let report = verify_variant(
            "vortex-turboquant",
            &tq_prep.array,
            &prepared.query,
            &baseline_scores,
            VerificationKind::Lossy,
            &SESSION,
        )
        .expect("TurboQuant verification should not error");
        assert!(
            report.passed,
            "TurboQuant drift {:.4} exceeds lossy tolerance {:.4}",
            report.max_abs_diff,
            report.tolerance()
        );
    }
}
