// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Recall@K quality measurement for lossy vector-search variants.
//!
//! This module computes the fraction of the true top-K nearest neighbours that a
//! lossy encoding (today just TurboQuant) recovers, using the uncompressed Vortex
//! scan as the local ground truth. Recall is averaged over a small number of sampled
//! query rows.
//!
//! This is explicitly a *relative* recall — we compare TurboQuant-retrieved neighbours
//! against the neighbours that the *same* cosine-similarity expression finds in the
//! uncompressed scan, not against VectorDBBench's shipped `neighbors.parquet`. Comparing
//! against external ground truth would require an index (which Vortex doesn't have) and
//! is structurally out of scope for a file-format benchmark.

use anyhow::Result;
use vortex::array::ArrayRef;
use vortex::utils::aliases::hash_set::HashSet;

use crate::extract_query_row;
use crate::verify::compute_cosine_scores;

/// Size of the neighbour set we compare. 10 is the standard VectorDBBench default.
pub const DEFAULT_TOP_K: usize = 10;

/// Compute recall@K for the lossy `compressed` variant against the `uncompressed`
/// ground-truth variant, averaged over `num_queries` sampled query rows. Uses the
/// global [`vortex_bench::SESSION`] for all executions.
///
/// Query selection is deterministic: rows are picked uniformly across the dataset at
/// `step = uncompressed.len() / num_queries` intervals. This keeps the result stable
/// across runs and avoids needing to thread a PRNG seed into the benchmark CLI.
pub fn measure_recall_at_k(
    uncompressed: &ArrayRef,
    compressed: &ArrayRef,
    num_queries: usize,
    top_k: usize,
) -> Result<f64> {
    assert!(
        num_queries > 0,
        "measure_recall_at_k requires num_queries > 0"
    );
    assert!(top_k > 0, "measure_recall_at_k requires top_k > 0");
    let num_rows = uncompressed.len();
    assert_eq!(
        compressed.len(),
        num_rows,
        "uncompressed and compressed arrays must have the same row count"
    );
    assert!(num_rows >= top_k, "dataset must have at least top_k rows");

    let step = (num_rows / num_queries).max(1);

    let mut total_hits: usize = 0;
    let mut total_checked: usize = 0;

    for q in 0..num_queries {
        let row = (q * step).min(num_rows - 1);
        let query = extract_query_row(uncompressed, row)?;

        let gt_scores = compute_cosine_scores(uncompressed, &query)?;
        let truth = top_k_indices(&gt_scores, top_k);

        let lossy_scores = compute_cosine_scores(compressed, &query)?;
        let lossy = top_k_indices(&lossy_scores, top_k);

        let truth_set: HashSet<usize> = truth.iter().copied().collect();
        total_hits += lossy.iter().filter(|idx| truth_set.contains(*idx)).count();
        total_checked += top_k;
    }

    Ok(total_hits as f64 / total_checked as f64)
}

/// Return the indices of the top-K highest scores, stable-sorted descending.
///
/// Uses `f32::total_cmp` for a NaN-safe total order — `partial_cmp` would panic on
/// NaN, and `partial_cmp(...).unwrap_or(Ordering::Equal)` would put NaNs at
/// arbitrary positions. `total_cmp` gives NaNs a well-defined (but meaningless) sort
/// slot, which lets the function be robust against accidental NaN inputs without
/// silently hiding them.
fn top_k_indices(scores: &[f32], top_k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| scores[b].total_cmp(&scores[a]));
    idx.truncate(top_k);
    idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Variant;
    use crate::extract_query_row;
    use crate::prepare_variant;
    use crate::test_utils::synthetic_vector;

    #[test]
    fn top_k_indices_handles_nan_without_panicking() {
        // `partial_cmp` panics on NaN (well, returns None, which was silently swallowed
        // before). `total_cmp` gives NaN a well-defined slot, so the sort doesn't
        // panic and doesn't produce arbitrary orderings for non-NaN elements.
        let scores = [0.9f32, f32::NAN, 0.7, 0.5, f32::NAN];
        let top = top_k_indices(&scores, 3);
        assert_eq!(top.len(), 3);
        // The finite values 0.9, 0.7, 0.5 should still rank in the right order
        // relative to each other — NaNs sort somewhere, but the finite ordering is
        // preserved because `total_cmp` is a total order.
        let finite_positions: Vec<usize> = top
            .iter()
            .copied()
            .filter(|&i| !scores[i].is_nan())
            .collect();
        assert!(
            finite_positions
                .windows(2)
                .all(|w| scores[w[0]] >= scores[w[1]]),
            "finite scores should still be in descending order"
        );
    }

    #[test]
    fn uncompressed_has_perfect_self_recall() {
        let dim = 128u32;
        let num_rows = 64usize;
        let uncompressed = synthetic_vector(dim, num_rows, 0xC0FFEE);

        let recall = measure_recall_at_k(&uncompressed, &uncompressed, 4, 10).unwrap();
        assert!(
            (recall - 1.0).abs() < 1e-9,
            "self-recall must be 1.0, got {recall}"
        );
    }

    #[test]
    fn turboquant_recall_is_reasonable_for_synthetic_data() {
        let dim = 128u32;
        let num_rows = 64usize;
        let uncompressed = synthetic_vector(dim, num_rows, 0xC0FFEE);

        // `measure_recall_at_k` doesn't need the PreparedDataset's `query` field —
        // it derives queries internally via `extract_query_row` on `uncompressed`.
        // Construct just enough of a `PreparedDataset` to pass to `prepare_variant`.
        let prepared = crate::PreparedDataset {
            name: "synthetic".to_string(),
            uncompressed: uncompressed.clone(),
            query: extract_query_row(&uncompressed, 0).unwrap(),
            parquet_bytes: 0,
        };

        let tq_prep = prepare_variant(&prepared, Variant::VortexTurboQuant).unwrap();

        // With only 64 random rows, recall@10 won't be 1.0 but it should be well
        // above chance (10/64 ≈ 0.156). The test asserts a loose lower bound to catch
        // total regressions without being flaky on distribution noise.
        let recall = measure_recall_at_k(&uncompressed, &tq_prep.array, 4, 10).unwrap();
        assert!(
            recall >= 0.3,
            "TurboQuant recall@10 on 64×128 synthetic data should be ≥0.3, got {recall}",
        );
    }
}
