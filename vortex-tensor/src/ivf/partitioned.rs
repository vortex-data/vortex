// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! IVF-partitioned data for efficient cluster-based reads.
//!
//! Given an IVF index, this module reorders data by cluster assignment so that all vectors
//! in the same cluster are contiguous. This enables efficient I/O: when probing a subset
//! of clusters, we only read the contiguous ranges of rows belonging to those clusters.
//!
//! # Data Organization
//!
//! After partitioning, the data is organized as:
//!
//! ```text
//! [cluster 0 vectors] [cluster 1 vectors] ... [cluster K-1 vectors]
//! ```
//!
//! The [`IvfPartitionedIndex`] stores the cluster boundaries (offsets) and a mapping from
//! the sorted position back to the original row index.

use vortex_error::VortexResult;

use super::IvfIndex;

/// An IVF index augmented with cluster boundary information for efficient range-based reads.
///
/// Created via [`IvfPartitionedIndex::from_index`], this struct contains everything needed
/// to identify which row ranges to read for a given set of probe clusters.
#[derive(Clone, Debug)]
pub struct IvfPartitionedIndex {
    /// The underlying IVF index (centroids and assignments).
    index: IvfIndex,
    /// Permutation: `permutation[sorted_pos] = original_row_idx`.
    /// Rows are sorted by cluster assignment.
    permutation: Vec<u32>,
    /// Cluster offsets: `cluster_offsets[i]` is the starting position (in the sorted order)
    /// of cluster `i`. `cluster_offsets[K]` is the total number of rows.
    cluster_offsets: Vec<u64>,
}

impl IvfPartitionedIndex {
    /// Build a partitioned index from an existing IVF index.
    ///
    /// Computes the permutation that sorts rows by cluster and the cluster boundary offsets.
    pub fn from_index(index: IvfIndex) -> Self {
        let num_vectors = index.num_vectors();
        let num_clusters = index.num_clusters();

        // Count the number of rows per cluster.
        let mut counts = vec![0u64; num_clusters];
        for &assignment in index.assignments() {
            counts[assignment as usize] += 1;
        }

        // Compute prefix sums for cluster offsets.
        let mut cluster_offsets = vec![0u64; num_clusters + 1];
        for idx in 0..num_clusters {
            cluster_offsets[idx + 1] = cluster_offsets[idx] + counts[idx];
        }

        // Build the permutation: for each row, place it in the next available slot of its cluster.
        let mut cursor = cluster_offsets[..num_clusters].to_vec();
        let mut permutation = vec![0u32; num_vectors];
        for (row_idx, &assignment) in index.assignments().iter().enumerate() {
            #[expect(
                clippy::cast_possible_truncation,
                reason = "cursor values and row indices fit in usize/u32 for realistic datasets"
            )]
            {
                let pos = cursor[assignment as usize] as usize;
                permutation[pos] = row_idx as u32;
            }
            cursor[assignment as usize] += 1;
        }

        Self {
            index,
            permutation,
            cluster_offsets,
        }
    }

    /// Returns the underlying IVF index.
    pub fn index(&self) -> &IvfIndex {
        &self.index
    }

    /// Returns the permutation from sorted position to original row index.
    pub fn permutation(&self) -> &[u32] {
        &self.permutation
    }

    /// Returns the cluster boundary offsets.
    ///
    /// `cluster_offsets[i]` is the starting row (in sorted order) of cluster `i`.
    /// Cluster `i` spans `cluster_offsets[i]..cluster_offsets[i+1]`.
    pub fn cluster_offsets(&self) -> &[u64] {
        &self.cluster_offsets
    }

    /// Given a set of probe clusters, return the row ranges (in sorted order) to read.
    ///
    /// Each range represents a contiguous block of rows belonging to a single cluster.
    /// The ranges are sorted by their start position.
    pub fn probe_ranges(&self, probed_clusters: &[usize]) -> Vec<std::ops::Range<u64>> {
        let mut ranges: Vec<std::ops::Range<u64>> = probed_clusters
            .iter()
            .filter(|&&c| c < self.index.num_clusters())
            .map(|&c| self.cluster_offsets[c]..self.cluster_offsets[c + 1])
            .filter(|r| !r.is_empty())
            .collect();
        ranges.sort_by_key(|r| r.start);
        ranges
    }

    /// Given a query vector, probe the index and return the row ranges to read.
    pub fn query_ranges(
        &self,
        query: &[f32],
        nprobes: usize,
    ) -> VortexResult<Vec<std::ops::Range<u64>>> {
        let probed = self.index.probe(query, nprobes)?;
        Ok(self.probe_ranges(&probed))
    }

    /// Returns the total fraction of rows that would be read for the given probe ranges.
    pub fn selectivity(&self, ranges: &[std::ops::Range<u64>]) -> f64 {
        let total = self.index.num_vectors() as f64;
        if total == 0.0 {
            return 0.0;
        }
        let selected: u64 = ranges.iter().map(|r| r.end - r.start).sum();
        selected as f64 / total
    }
}

#[cfg(test)]
mod tests {
    use vortex_error::VortexResult;

    use super::*;
    use crate::ivf::IvfBuildConfig;
    use crate::ivf::tests::generate_clustered_vectors;
    use crate::ivf::tests::normalize_vectors;

    #[test]
    fn partitioned_index_basic() -> VortexResult<()> {
        // 8 vectors of dim=3 in 2 clusters.
        #[rustfmt::skip]
        let vectors: Vec<f32> = vec![
            0.95, 0.05, 0.0,  // cluster A
            0.0, 0.95, 0.05,  // cluster B
            1.00, 0.00, 0.0,  // cluster A
            0.0, 1.00, 0.00,  // cluster B
            0.90, 0.10, 0.0,  // cluster A
            0.0, 0.90, 0.10,  // cluster B
        ];
        let config = IvfBuildConfig {
            num_clusters: 2,
            max_iterations: 20,
            seed: 42,
        };
        let index = IvfIndex::build(&vectors, 3, &config)?;
        let partitioned = IvfPartitionedIndex::from_index(index);

        // Check that offsets cover all rows.
        let offsets = partitioned.cluster_offsets();
        assert_eq!(offsets.len(), 3); // 2 clusters + sentinel
        assert_eq!(offsets[0], 0);
        assert_eq!(*offsets.last().unwrap(), 6);

        // Each cluster should have 3 rows.
        assert_eq!(offsets[1] - offsets[0], 3);
        assert_eq!(offsets[2] - offsets[1], 3);

        // Permutation should map all 6 rows.
        let perm = partitioned.permutation();
        assert_eq!(perm.len(), 6);
        let mut sorted_perm: Vec<u32> = perm.to_vec();
        sorted_perm.sort();
        assert_eq!(sorted_perm, vec![0, 1, 2, 3, 4, 5]);

        Ok(())
    }

    #[test]
    fn probe_ranges_correct() -> VortexResult<()> {
        let mut vectors = generate_clustered_vectors(100, 16, 4, 42);
        normalize_vectors(&mut vectors, 16);

        let config = IvfBuildConfig {
            num_clusters: 4,
            max_iterations: 20,
            seed: 42,
        };
        let index = IvfIndex::build(&vectors, 16, &config)?;
        let partitioned = IvfPartitionedIndex::from_index(index);

        // Probe 2 clusters.
        let ranges = partitioned.query_ranges(&vectors[0..16], 2)?;
        assert!(!ranges.is_empty());
        assert!(ranges.len() <= 2);

        // Ranges should not overlap and should be sorted.
        for i in 1..ranges.len() {
            assert!(ranges[i].start >= ranges[i - 1].end);
        }

        // Selectivity should be less than 1.0 (we're probing 2 of 4 clusters).
        let selectivity = partitioned.selectivity(&ranges);
        assert!(selectivity > 0.0);
        assert!(selectivity < 1.0);

        Ok(())
    }

    #[test]
    fn probe_all_gives_full_selectivity() -> VortexResult<()> {
        let vectors = generate_clustered_vectors(50, 8, 3, 42);
        let config = IvfBuildConfig {
            num_clusters: 3,
            max_iterations: 20,
            seed: 42,
        };
        let index = IvfIndex::build(&vectors, 8, &config)?;
        let partitioned = IvfPartitionedIndex::from_index(index);

        let ranges = partitioned.query_ranges(&vectors[0..8], 100)?; // probe all
        let selectivity = partitioned.selectivity(&ranges);
        assert!((selectivity - 1.0).abs() < 1e-10);

        Ok(())
    }
}
