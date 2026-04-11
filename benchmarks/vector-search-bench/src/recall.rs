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
use vortex::array::IntoArray;
use vortex::array::VortexSessionExecute;
use vortex::array::arrays::PrimitiveArray;
use vortex::array::arrays::extension::ExtensionArrayExt;
use vortex::error::VortexExpect;
use vortex::session::VortexSession;
use vortex::utils::aliases::hash_set::HashSet;
use vortex_tensor::scalar_fns::cosine_similarity::CosineSimilarity;
use vortex_tensor::vector_search::build_constant_query_vector;

/// Size of the neighbour set we compare. 10 is the standard VectorDBBench default.
pub const DEFAULT_TOP_K: usize = 10;

/// Compute recall@K for the lossy `compressed` variant against the `uncompressed`
/// ground-truth variant, averaged over `num_queries` sampled query rows.
///
/// Query selection is deterministic: rows are picked uniformly across the dataset at
/// `step = uncompressed.len() / num_queries` intervals. This keeps the result stable
/// across runs and avoids needing to thread a PRNG seed into the benchmark CLI.
pub fn measure_recall_at_k(
    uncompressed: &ArrayRef,
    compressed: &ArrayRef,
    num_queries: usize,
    top_k: usize,
    session: &VortexSession,
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
        let query = extract_query_row(uncompressed, row, session)?;

        let gt_scores = score_all_rows(uncompressed, &query, session)?;
        let truth = top_k_indices(&gt_scores, top_k);

        let lossy_scores = score_all_rows(compressed, &query, session)?;
        let lossy = top_k_indices(&lossy_scores, top_k);

        let truth_set: HashSet<usize> = truth.iter().copied().collect();
        total_hits += lossy.iter().filter(|idx| truth_set.contains(*idx)).count();
        total_checked += top_k;
    }

    Ok(total_hits as f64 / total_checked as f64)
}

fn extract_query_row(
    vector_ext: &ArrayRef,
    row: usize,
    session: &VortexSession,
) -> Result<Vec<f32>> {
    use anyhow::Context;
    use vortex::array::arrays::Extension;
    use vortex::array::arrays::FixedSizeListArray;
    use vortex::array::arrays::fixed_size_list::FixedSizeListArrayExt;

    let mut ctx = session.create_execution_ctx();
    let ext = vector_ext
        .as_opt::<Extension>()
        .context("extract_query_row expects an Extension<Vector> array")?;
    let fsl: FixedSizeListArray = ext.storage_array().clone().execute(&mut ctx)?;

    let dim_usize = match fsl.dtype() {
        vortex::dtype::DType::FixedSizeList(_, dim, _) => *dim as usize,
        other => anyhow::bail!("expected FixedSizeList storage, got {other}"),
    };

    let elements: PrimitiveArray = fsl.elements().clone().execute(&mut ctx)?;
    let slice = elements.as_slice::<f32>();
    let start = row * dim_usize;
    Ok(slice[start..start + dim_usize].to_vec())
}

fn score_all_rows(data: &ArrayRef, query: &[f32], session: &VortexSession) -> Result<Vec<f32>> {
    let num_rows = data.len();
    let query_vec = build_constant_query_vector(query, num_rows)?;
    let cosine = CosineSimilarity::try_new_array(data.clone(), query_vec, num_rows)
        .vortex_expect("cosine similarity accepts matching Vector inputs")
        .into_array();

    let mut ctx = session.create_execution_ctx();
    let scores: PrimitiveArray = cosine.execute(&mut ctx)?;
    Ok(scores.as_slice::<f32>().to_vec())
}

/// Return the indices of the top-K highest scores, stable-sorted descending.
fn top_k_indices(scores: &[f32], top_k: usize) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..scores.len()).collect();
    idx.sort_by(|&a, &b| {
        scores[b]
            .partial_cmp(&scores[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(top_k);
    idx
}

#[cfg(test)]
mod tests {
    use vortex_bench::SESSION;

    use super::*;
    use crate::Variant;
    use crate::prepare_variant;
    use crate::test_utils::synthetic_vector;

    #[test]
    fn uncompressed_has_perfect_self_recall() {
        let dim = 128u32;
        let num_rows = 64usize;
        let uncompressed = synthetic_vector(dim, num_rows, 0xC0FFEE);

        let recall = measure_recall_at_k(&uncompressed, &uncompressed, 4, 10, &SESSION).unwrap();
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

        let prepared = crate::PreparedDataset {
            name: "synthetic".to_string(),
            uncompressed: uncompressed.clone(),
            query: vec![],
            parquet_bytes: 0,
        };

        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let (tq_array, _) = rt
            .block_on(prepare_variant(
                &prepared,
                Variant::VortexTurboQuant,
                &SESSION,
            ))
            .unwrap();

        // With only 64 random rows, recall@10 won't be 1.0 but it should be well
        // above chance (10/64 ≈ 0.156). The test asserts a loose lower bound to catch
        // total regressions without being flaky on distribution noise.
        let recall = measure_recall_at_k(&uncompressed, &tq_array, 4, 10, &SESSION).unwrap();
        assert!(
            recall >= 0.3,
            "TurboQuant recall@10 on 64×128 synthetic data should be ≥0.3, got {recall}",
        );
    }
}
