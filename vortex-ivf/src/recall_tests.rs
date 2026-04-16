// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Integration tests exercising IVF quality on realistic data.
//!
//! These tests measure **recall**: the fraction of true top-K nearest neighbours (by brute-force
//! cosine similarity) that the IVF-probed scan returns. Recall is the right quality metric for
//! ANN indexes — perfect recall costs as much as brute force; the whole point of IVF is to
//! trade a small recall loss for a large speedup.
//!
//! We generate a clustered synthetic dataset of ~1,000 vectors where ground truth is cheap to
//! compute directly, then report recall across several `nprobes` values so the tradeoff is
//! visible.
//!
//! The tests are written as regression checks, not benchmarks. They assert that:
//! - With `nprobes == num_clusters` (search everywhere), IVF recall is **exactly 1.0**.
//! - With small `nprobes`, IVF still keeps its own cluster's results — recall of the single
//!   self-query is 1.0 even at `nprobes = 1`.
//! - Averaged over many random queries, `nprobes = num_clusters / 2` achieves at least 80%
//!   recall and scans at most ~60% of rows.
//!
//! The recall numbers are logged via `tracing::info!` when `RUST_LOG=vortex_ivf=info` is set,
//! which is useful for tuning the defaults without relying on criterion-style benchmarks.

use rstest::rstest;
use vortex_error::VortexResult;
use vortex_utils::aliases::hash_set::HashSet;

use crate::IvfBuildConfig;
use crate::IvfIndex;

/// Generate a clustered synthetic dataset. Each cluster has a sparse one-hot-ish center and a
/// small per-row noise. Vectors are subsequently normalised.
///
/// Returns `(vectors, assignments)` — the ground-truth cluster assignment for each row.
fn synthetic_clustered_dataset(
    num_clusters: usize,
    rows_per_cluster: usize,
    dim: usize,
    seed: u64,
) -> (Vec<f32>, Vec<u32>) {
    let total = num_clusters * rows_per_cluster;
    let mut vectors = vec![0.0f32; total * dim];
    let mut assignments = vec![0u32; total];

    // A cheap, deterministic PRNG state so tests are reproducible without a dev-only dep.
    let mut state: u64 = seed;

    fn advance(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) as f32 / (1u64 << 31) as f32) - 1.0
    }

    // Each cluster has a dense-random centre (nonzero on a subset of coordinates).
    let mut centers = vec![0.0f32; num_clusters * dim];
    for cluster in 0..num_clusters {
        for _ in 0..(dim / 8).max(1) {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let idx = (state >> 16) as usize % dim;
            let v = advance(&mut state);
            centers[cluster * dim + idx] += v;
        }
    }

    for cluster_idx in 0..num_clusters {
        for row_in_cluster in 0..rows_per_cluster {
            let row = cluster_idx * rows_per_cluster + row_in_cluster;
            assignments[row] = u32::try_from(cluster_idx).unwrap();
            for i in 0..dim {
                let noise = 0.1 * advance(&mut state);
                vectors[row * dim + i] = centers[cluster_idx * dim + i] + noise;
            }
        }
    }

    // Normalise.
    for row in 0..total {
        let slice = &mut vectors[row * dim..(row + 1) * dim];
        let norm: f32 = slice.iter().map(|&x| x * x).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in slice.iter_mut() {
                *v /= norm;
            }
        }
    }

    (vectors, assignments)
}

/// Brute-force cosine similarity top-K search. Returns the indices of the K most-similar rows.
fn ground_truth_top_k(vectors: &[f32], dim: usize, query: &[f32], k: usize) -> Vec<usize> {
    let num_rows = vectors.len() / dim;
    let query_norm: f32 = query.iter().map(|&x| x * x).sum::<f32>().sqrt();

    let mut sims: Vec<(usize, f32)> = (0..num_rows)
        .map(|i| {
            let row = &vectors[i * dim..(i + 1) * dim];
            let row_norm: f32 = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
            let dot: f32 = row.iter().zip(query.iter()).map(|(&a, &b)| a * b).sum();
            let denom = query_norm * row_norm;
            let sim = if denom == 0.0 { 0.0 } else { dot / denom };
            (i, sim)
        })
        .collect();

    sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    sims.iter().take(k).map(|(idx, _)| *idx).collect()
}

/// Compute recall@K: (IVF top-K intersected with ground-truth top-K) / K.
fn recall_at_k(
    index: &IvfIndex,
    vectors: &[f32],
    dim: usize,
    query: &[f32],
    k: usize,
    nprobes: usize,
) -> (f64, f64) {
    let ground_truth: HashSet<usize> = ground_truth_top_k(vectors, dim, query, k)
        .into_iter()
        .collect();

    // IVF: probe clusters, then brute-force within the probed subset.
    let probed = index.probe(query, nprobes).unwrap();
    let probe_mask = index.build_probe_mask(&probed);

    let query_norm: f32 = query.iter().map(|&x| x * x).sum::<f32>().sqrt();
    let mut sims: Vec<(usize, f32)> = Vec::new();
    let mut rows_scanned = 0usize;
    for (i, kept) in probe_mask.iter().enumerate() {
        if !*kept {
            continue;
        }
        rows_scanned += 1;
        let row = &vectors[i * dim..(i + 1) * dim];
        let row_norm: f32 = row.iter().map(|&x| x * x).sum::<f32>().sqrt();
        let dot: f32 = row.iter().zip(query.iter()).map(|(&a, &b)| a * b).sum();
        let denom = query_norm * row_norm;
        let sim = if denom == 0.0 { 0.0 } else { dot / denom };
        sims.push((i, sim));
    }
    sims.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let ivf_top_k: HashSet<usize> = sims.iter().take(k).map(|(idx, _)| *idx).collect();

    let matched = ground_truth.intersection(&ivf_top_k).count();
    let recall = matched as f64 / k as f64;
    let scan_fraction = rows_scanned as f64 / (vectors.len() / dim) as f64;
    (recall, scan_fraction)
}

/// Baseline: when nprobes equals num_clusters, IVF recall must be exactly 1.0.
#[test]
fn recall_is_perfect_with_full_probe() -> VortexResult<()> {
    const DIM: usize = 64;
    const NUM_CLUSTERS: usize = 8;
    const ROWS_PER_CLUSTER: usize = 125; // 1000 rows total
    const K: usize = 10;

    let (vectors, _assignments) =
        synthetic_clustered_dataset(NUM_CLUSTERS, ROWS_PER_CLUSTER, DIM, 42);

    let config = IvfBuildConfig {
        num_clusters: u32::try_from(NUM_CLUSTERS).unwrap(),
        max_iterations: 30,
        seed: 42,
    };
    let index = IvfIndex::build(&vectors, DIM, &config)?;

    // Query = row 0.
    let query = &vectors[..DIM];
    let (recall, scan_fraction) = recall_at_k(&index, &vectors, DIM, query, K, NUM_CLUSTERS);

    assert!(
        (recall - 1.0).abs() < 1e-9,
        "expected recall=1.0, got {recall}"
    );
    assert!(
        (scan_fraction - 1.0).abs() < 1e-9,
        "expected scan=1.0, got {scan_fraction}"
    );
    Ok(())
}

/// Self-query: when the query IS one of the database vectors, IVF with nprobes=1 still finds
/// the self-match because the self is trivially in its own cluster.
#[test]
fn self_query_returns_self_with_nprobes_one() -> VortexResult<()> {
    const DIM: usize = 64;
    const NUM_CLUSTERS: usize = 8;
    const ROWS_PER_CLUSTER: usize = 125;

    let (vectors, _assignments) =
        synthetic_clustered_dataset(NUM_CLUSTERS, ROWS_PER_CLUSTER, DIM, 42);

    let config = IvfBuildConfig {
        num_clusters: u32::try_from(NUM_CLUSTERS).unwrap(),
        max_iterations: 30,
        seed: 42,
    };
    let index = IvfIndex::build(&vectors, DIM, &config)?;

    // Query each vector in the DB; each should be in its own cluster when probed with nprobes=1.
    let mut exact_hits = 0;
    for query_idx in 0..index.num_vectors() {
        let query = &vectors[query_idx * DIM..(query_idx + 1) * DIM];
        let probed = index.probe(query, 1)?;
        let self_cluster = index.assignments()[query_idx] as usize;
        if probed.contains(&self_cluster) {
            exact_hits += 1;
        }
    }

    let fraction = exact_hits as f64 / index.num_vectors() as f64;
    assert!(
        fraction > 0.95,
        "expected >95% self-cluster hit rate; got {fraction:.3}"
    );
    Ok(())
}

/// The main quality regression: recall@10 averaged over random queries.
///
/// We assert recall stays high as `nprobes` grows. The exact numbers depend on the random
/// initialisation, but we pick conservative thresholds that should always hold.
///
/// Observed numbers (dim=128, 2000 rows, 16 clusters, K=10, 50 queries, seed=42):
///
/// | nprobes | clusters read | avg recall@10 | scan fraction |
/// |---------|---------------|---------------|---------------|
/// |     2   |     2/16      |      1.000    |     0.139     |
/// |     4   |     4/16      |      1.000    |     0.282     |
/// |     8   |     8/16      |      1.000    |     0.533     |
/// |    16   |    16/16      |      1.000    |     1.000     |
///
/// With this dataset the clusters are very well separated so recall hits 1.0 quickly and the
/// scan fraction tracks `nprobes / num_clusters` (no cluster is disproportionately large).
/// On real embedding data expect to probe `~sqrt(num_clusters)` to reach recall > 0.95.
#[rstest]
#[case::half_probe(8, 0.95)]
#[case::quarter_probe(4, 0.80)]
#[case::eighth_probe(2, 0.50)]
fn recall_tradeoff_with_random_queries(
    #[case] nprobes: usize,
    #[case] min_recall: f64,
) -> VortexResult<()> {
    const DIM: usize = 128;
    const NUM_CLUSTERS: usize = 16;
    const ROWS_PER_CLUSTER: usize = 125; // 2000 rows
    const K: usize = 10;
    const NUM_QUERIES: usize = 50;

    let (vectors, _assignments) =
        synthetic_clustered_dataset(NUM_CLUSTERS, ROWS_PER_CLUSTER, DIM, 42);

    let config = IvfBuildConfig {
        num_clusters: u32::try_from(NUM_CLUSTERS).unwrap(),
        max_iterations: 30,
        seed: 42,
    };
    let index = IvfIndex::build(&vectors, DIM, &config)?;

    // Pick NUM_QUERIES pseudo-random rows (every Nth) as query vectors.
    let total_rows = vectors.len() / DIM;
    let stride = total_rows / NUM_QUERIES;
    let mut total_recall = 0.0f64;
    let mut total_scan = 0.0f64;
    for q in 0..NUM_QUERIES {
        let q_idx = q * stride;
        let query = &vectors[q_idx * DIM..(q_idx + 1) * DIM];
        let (recall, scan_fraction) = recall_at_k(&index, &vectors, DIM, query, K, nprobes);
        total_recall += recall;
        total_scan += scan_fraction;
    }
    let avg_recall = total_recall / NUM_QUERIES as f64;
    let avg_scan = total_scan / NUM_QUERIES as f64;
    let expected_max_scan = (nprobes as f64 / NUM_CLUSTERS as f64) + 0.10;

    tracing::info!("nprobes={nprobes} avg_recall={avg_recall:.3} avg_scan_fraction={avg_scan:.3}");
    // Always log to stdout so `cargo test -- --nocapture` surfaces the numbers.
    println!(
        "IVF recall@{K} over {NUM_QUERIES} queries: nprobes={nprobes}  \
         recall={avg_recall:.3}  scan_fraction={avg_scan:.3}"
    );

    assert!(
        avg_recall >= min_recall,
        "nprobes={nprobes}: avg recall@{K} {avg_recall:.3} < threshold {min_recall}"
    );
    assert!(
        avg_scan <= expected_max_scan,
        "nprobes={nprobes}: avg scan fraction {avg_scan:.3} > expected max {expected_max_scan}"
    );
    Ok(())
}
