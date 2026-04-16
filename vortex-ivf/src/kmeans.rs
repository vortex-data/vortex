// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! K-means clustering implementation for IVF index construction.
//!
//! Uses k-means++ initialization for better centroid placement, then runs
//! Lloyd's algorithm for refinement. All computation is in f32. Vectors
//! are expected in row-major layout: a flat `[n * dim]` slice.

/// A minimal deterministic PRNG used for k-means++ initialization.
///
/// This is a SplitMix64 implementation that matches the one used in other parts of Vortex
/// (see `vortex-tensor::scalar_fns::sorf_transform::splitmix64`). It is duplicated here
/// to avoid a public dependency on that private module.
struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }
}

/// Result of k-means clustering.
pub(super) struct KMeansResult {
    /// Final centroids as a flat `[k * dim]` array in row-major order.
    pub centroids: Vec<f32>,
    /// Cluster assignment for each input vector (values in `0..k`).
    pub assignments: Vec<u32>,
}

/// Run k-means clustering on the given vectors.
///
/// `vectors` is a flat `[n * dim]` slice. `num_clusters` is the number of clusters.
/// `max_iter` is the maximum number of Lloyd iterations.
/// `seed` is used for k-means++ initialization.
pub(super) fn kmeans(
    vectors: &[f32],
    dim: usize,
    num_clusters: usize,
    max_iter: usize,
    seed: u64,
) -> KMeansResult {
    debug_assert!(dim > 0);
    debug_assert!(num_clusters > 0);
    debug_assert!(vectors.len().is_multiple_of(dim));

    let num_vectors = vectors.len() / dim;
    debug_assert!(num_clusters <= num_vectors);

    // Initialize centroids with k-means++.
    let mut centroids = kmeans_pp_init(vectors, dim, num_clusters, seed);
    let mut assignments = vec![0u32; num_vectors];

    for _iteration in 0..max_iter {
        // Assignment step: assign each vector to its nearest centroid.
        let changed = assign(vectors, dim, &centroids, num_clusters, &mut assignments);

        // Update step: recompute centroids as the mean of assigned vectors.
        update_centroids(vectors, dim, num_clusters, &assignments, &mut centroids);

        // If no assignments changed, we've converged.
        if !changed {
            break;
        }
    }

    KMeansResult {
        centroids,
        assignments,
    }
}

/// K-means++ initialization: pick centroids that are spread out.
///
/// 1. Pick the first centroid uniformly at random.
/// 2. For each subsequent centroid, pick a vector with probability proportional
///    to its squared distance from the nearest existing centroid.
fn kmeans_pp_init(vectors: &[f32], dim: usize, num_clusters: usize, seed: u64) -> Vec<f32> {
    let num_vectors = vectors.len() / dim;
    let mut rng = SplitMix64::new(seed);
    let mut centroids = Vec::with_capacity(num_clusters * dim);

    // Pick first centroid uniformly at random.
    #[expect(
        clippy::cast_possible_truncation,
        reason = "num_vectors always fits in usize; the modulo reduces the u64 range"
    )]
    let first_idx = (rng.next_u64() % num_vectors as u64) as usize;
    centroids.extend_from_slice(&vectors[first_idx * dim..(first_idx + 1) * dim]);

    // Squared distances from each vector to its nearest centroid.
    let mut min_dists = vec![f32::MAX; num_vectors];

    for centroid_idx in 1..num_clusters {
        // Update min distances with the last centroid added.
        let last_centroid = &centroids[(centroid_idx - 1) * dim..centroid_idx * dim];
        for idx in 0..num_vectors {
            let vec_slice = &vectors[idx * dim..(idx + 1) * dim];
            let dist = squared_l2_distance(vec_slice, last_centroid);
            if dist < min_dists[idx] {
                min_dists[idx] = dist;
            }
        }

        // Pick next centroid with probability proportional to squared distance.
        let total: f64 = min_dists.iter().map(|&dist| dist as f64).sum();
        if total <= 0.0 {
            // All remaining vectors are identical to existing centroids.
            // Just pick sequentially to fill remaining slots.
            let next_idx = centroid_idx % num_vectors;
            centroids.extend_from_slice(&vectors[next_idx * dim..(next_idx + 1) * dim]);
            continue;
        }

        // Use the PRNG to sample proportionally to squared distances.
        let threshold = (rng.next_u64() as f64 / u64::MAX as f64) * total;
        let mut cumulative = 0.0f64;
        let mut chosen = num_vectors - 1; // fallback to last
        for idx in 0..num_vectors {
            cumulative += min_dists[idx] as f64;
            if cumulative >= threshold {
                chosen = idx;
                break;
            }
        }

        centroids.extend_from_slice(&vectors[chosen * dim..(chosen + 1) * dim]);
    }

    centroids
}

/// Assign each vector to its nearest centroid. Returns true if any assignment changed.
fn assign(
    vectors: &[f32],
    dim: usize,
    centroids: &[f32],
    num_clusters: usize,
    assignments: &mut [u32],
) -> bool {
    let num_vectors = vectors.len() / dim;
    let mut changed = false;

    for row_idx in 0..num_vectors {
        let vec_slice = &vectors[row_idx * dim..(row_idx + 1) * dim];
        let mut best_dist = f32::MAX;
        let mut best_cluster = 0u32;

        for cluster_idx in 0..num_clusters {
            let centroid = &centroids[cluster_idx * dim..(cluster_idx + 1) * dim];
            let dist = squared_l2_distance(vec_slice, centroid);
            if dist < best_dist {
                best_dist = dist;
                #[expect(
                    clippy::cast_possible_truncation,
                    reason = "num_clusters <= u32::MAX by IvfBuildConfig validation"
                )]
                {
                    best_cluster = cluster_idx as u32;
                }
            }
        }

        if assignments[row_idx] != best_cluster {
            assignments[row_idx] = best_cluster;
            changed = true;
        }
    }

    changed
}

/// Recompute centroids as the mean of all vectors assigned to each cluster.
///
/// If a cluster has no assignments, its centroid is left unchanged.
fn update_centroids(
    vectors: &[f32],
    dim: usize,
    num_clusters: usize,
    assignments: &[u32],
    centroids: &mut [f32],
) {
    let num_vectors = vectors.len() / dim;

    // Accumulate sums and counts per cluster.
    let mut sums = vec![0.0f64; num_clusters * dim];
    let mut counts = vec![0u32; num_clusters];

    for row_idx in 0..num_vectors {
        let cluster = assignments[row_idx] as usize;
        counts[cluster] += 1;
        let vec_slice = &vectors[row_idx * dim..(row_idx + 1) * dim];
        let sum_slice = &mut sums[cluster * dim..(cluster + 1) * dim];
        for (dst, &src) in sum_slice.iter_mut().zip(vec_slice.iter()) {
            *dst += src as f64;
        }
    }

    // Update centroids. Leave empty clusters unchanged.
    for cluster_idx in 0..num_clusters {
        if counts[cluster_idx] > 0 {
            let count = counts[cluster_idx] as f64;
            let sum_slice = &sums[cluster_idx * dim..(cluster_idx + 1) * dim];
            let centroid_slice = &mut centroids[cluster_idx * dim..(cluster_idx + 1) * dim];
            #[expect(
                clippy::cast_possible_truncation,
                reason = "converting f64 mean back to f32 is intentional"
            )]
            for (dst, &src) in centroid_slice.iter_mut().zip(sum_slice.iter()) {
                *dst = (src / count) as f32;
            }
        }
    }
}

/// Squared L2 distance between two vectors of the same dimension.
#[inline]
fn squared_l2_distance(left: &[f32], right: &[f32]) -> f32 {
    debug_assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right.iter())
        .map(|(&lv, &rv)| {
            let diff = lv - rv;
            diff * diff
        })
        .sum()
}
