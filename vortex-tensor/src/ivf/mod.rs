// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! IVF (Inverted File) vector index for approximate nearest neighbor search.
//!
//! This module implements an IVF index that clusters vectors into K groups using k-means,
//! then at query time only searches the `nprobes` most promising clusters. This trades a
//! small amount of recall for a large speedup by avoiding brute-force comparison against
//! every vector.
//!
//! # Overview
//!
//! 1. **Build**: Run k-means on the database vectors to find K centroids. Assign each vector
//!    to its nearest centroid cluster.
//! 2. **Query**: Given a query vector, compute its similarity to all K centroids, select the
//!    top `nprobes` clusters, and search only the vectors in those clusters.
//!
//! # Integration with Vortex
//!
//! The IVF index works with [`Vector`](crate::vector::Vector) extension arrays. When data
//! is TurboQuant-compressed, the index operates on the compressed representation where
//! possible.
//!
//! The main entry points are:
//!
//! - [`IvfIndex::build`] to construct an index from vectors
//! - [`IvfIndex::probe`] to find which clusters to search
//! - [`IvfIndex::build_probe_mask`] to generate a boolean mask for row-level filtering

mod kmeans;
pub mod partitioned;
pub mod search;

#[cfg(test)]
mod tests;

use vortex_error::VortexResult;
use vortex_error::vortex_ensure;

/// Configuration for building an IVF index.
#[derive(Clone, Debug)]
pub struct IvfBuildConfig {
    /// Number of clusters (K). Must be >= 1.
    pub num_clusters: u32,
    /// Maximum number of k-means iterations.
    pub max_iterations: u32,
    /// Random seed for k-means initialization.
    pub seed: u64,
}

impl Default for IvfBuildConfig {
    fn default() -> Self {
        Self {
            num_clusters: 64,
            max_iterations: 20,
            seed: 42,
        }
    }
}

/// An IVF (Inverted File) index for approximate nearest neighbor search.
///
/// Contains K cluster centroids and the assignment of each database vector to a cluster.
/// At query time, only the `nprobes` clusters nearest to the query vector are searched,
/// providing a significant speedup over brute-force search at the cost of some recall.
#[derive(Clone, Debug)]
pub struct IvfIndex {
    /// Cluster centroids stored as a flat `[K * dim]` array in row-major order.
    centroids: Vec<f32>,
    /// The vector dimensionality.
    dim: usize,
    /// Number of clusters (K).
    num_clusters: usize,
    /// The cluster assignment for each row in the *original* (unsorted) order.
    assignments: Vec<u32>,
}

impl IvfIndex {
    /// Build an IVF index from a flat f32 vector array.
    ///
    /// `vectors` is a flat row-major `[num_vectors * dim]` buffer. Every contiguous
    /// group of `dim` values represents one vector.
    ///
    /// # Errors
    ///
    /// Returns an error if the input dimensions are inconsistent or if `num_clusters` is 0.
    pub fn build(vectors: &[f32], dim: usize, config: &IvfBuildConfig) -> VortexResult<Self> {
        vortex_ensure!(dim > 0, "IVF dimension must be > 0");
        vortex_ensure!(
            config.num_clusters >= 1,
            "IVF num_clusters must be >= 1, got {}",
            config.num_clusters
        );
        vortex_ensure!(
            vectors.len().is_multiple_of(dim),
            "vectors length {} is not a multiple of dim {}",
            vectors.len(),
            dim
        );

        let num_vectors = vectors.len() / dim;
        if num_vectors == 0 {
            return Ok(Self {
                centroids: vec![0.0; config.num_clusters as usize * dim],
                dim,
                num_clusters: config.num_clusters as usize,
                assignments: Vec::new(),
            });
        }

        // Clamp num_clusters to num_vectors (can't have more clusters than vectors).
        let k = (config.num_clusters as usize).min(num_vectors);

        let result = kmeans::kmeans(vectors, dim, k, config.max_iterations as usize, config.seed);

        Ok(Self {
            centroids: result.centroids,
            dim,
            num_clusters: k,
            assignments: result.assignments,
        })
    }

    /// Returns the cluster centroids as a flat `[K * dim]` slice in row-major order.
    pub fn centroids(&self) -> &[f32] {
        &self.centroids
    }

    /// Returns the dimensionality of the indexed vectors.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Returns the number of clusters (K).
    pub fn num_clusters(&self) -> usize {
        self.num_clusters
    }

    /// Returns the cluster assignment for each vector in the original order.
    pub fn assignments(&self) -> &[u32] {
        &self.assignments
    }

    /// Returns the number of indexed vectors.
    pub fn num_vectors(&self) -> usize {
        self.assignments.len()
    }

    /// Find the `nprobes` clusters whose centroids are most similar to the query vector
    /// (by cosine similarity).
    ///
    /// Returns the cluster indices sorted by descending similarity. If `nprobes` exceeds
    /// the number of clusters, all clusters are returned.
    pub fn probe(&self, query: &[f32], nprobes: usize) -> VortexResult<Vec<usize>> {
        vortex_ensure!(
            query.len() == self.dim,
            "query dimension {} does not match index dimension {}",
            query.len(),
            self.dim
        );

        let nprobes = nprobes.min(self.num_clusters);
        let query_norm = l2_norm(query);

        // Compute cosine similarity of query to each centroid.
        let mut similarities: Vec<(usize, f32)> = (0..self.num_clusters)
            .map(|i| {
                let centroid = &self.centroids[i * self.dim..(i + 1) * self.dim];
                let centroid_norm = l2_norm(centroid);
                let dot = dot_product(query, centroid);
                let denom = query_norm * centroid_norm;
                let sim = if denom == 0.0 { 0.0 } else { dot / denom };
                (i, sim)
            })
            .collect();

        // Sort by descending similarity and take the top nprobes.
        similarities
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(similarities.iter().take(nprobes).map(|(i, _)| *i).collect())
    }

    /// Build a boolean mask where `true` indicates the row belongs to one of the probed clusters.
    ///
    /// The mask has length equal to `self.num_vectors()`. Rows assigned to clusters in
    /// `probed_clusters` are marked `true`; all others are `false`.
    pub fn build_probe_mask(&self, probed_clusters: &[usize]) -> Vec<bool> {
        let mut cluster_set = vec![false; self.num_clusters];
        for &c in probed_clusters {
            if c < self.num_clusters {
                cluster_set[c] = true;
            }
        }

        self.assignments
            .iter()
            .map(|&a| cluster_set[a as usize])
            .collect()
    }

    /// Convenience: probe the index for the given query and return a boolean mask of rows to scan.
    ///
    /// Combines [`probe`](Self::probe) and [`build_probe_mask`](Self::build_probe_mask).
    pub fn query_mask(&self, query: &[f32], nprobes: usize) -> VortexResult<Vec<bool>> {
        let probed = self.probe(query, nprobes)?;
        Ok(self.build_probe_mask(&probed))
    }

    /// Returns the number of vectors in each cluster, indexed by cluster ID.
    pub fn cluster_sizes(&self) -> Vec<usize> {
        let mut sizes = vec![0usize; self.num_clusters];
        for &a in &self.assignments {
            sizes[a as usize] += 1;
        }
        sizes
    }
}

/// Compute the L2 norm of a vector.
fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|&x| x * x).sum::<f32>().sqrt()
}

/// Compute the dot product of two vectors.
fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}
